from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import io
import aiohttp
import random
from replicate import Replicate
from datetime import datetime, timedelta
import pandas as pd

from uuid import uuid4

import requests
import telegram.error
import xmltodict
from telegram import (
    BotCommandScopeAllGroupChats,
    Update,
    constants,
)
from telegram import (
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    InlineQueryResultArticle,
)
from telegram import InputTextMessageContent, BotCommand
from telegram.error import RetryAfter, TimedOut, BadRequest
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    InlineQueryHandler,
    CallbackQueryHandler,
    Application,
    ContextTypes,
    CallbackContext,
)

from pydub import AudioSegment
from PIL import Image

from utils import (
    is_group_chat,
    get_thread_id,
    message_text,
    wrap_with_indicator,
    split_into_chunks,
    edit_message_with_retry,
    get_stream_cutoff_values,
    is_allowed,
    is_admin,
    is_within_budget,
    get_reply_to_message_id,
    add_chat_request_to_usage_tracker,
    is_direct_result,
    handle_direct_result,
    cleanup_intermediate_files,
)
from openai_helper import OpenAIHelper, localized_text
from usage_tracker import UsageTracker
from db import DB


def mask_api_key(api_key):
    return api_key[:6] + "..." + api_key[-4:]


def model_keyboard(default_model: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    text="✅ GPT-3.5" if default_model == "gpt35" else "GPT-3.5",
                    callback_data="change_model_gpt35",
                ),
                InlineKeyboardButton(
                    text="✅ GPT-4 Turbo"
                    if default_model == "gpt4_turbo"
                    else "GPT-4 Turbo",
                    callback_data="change_model_gpt4_turbo",
                ),
            ]
        ]
    )


class ChatGPTTelegramBot:
    """
    Class representing a ChatGPT Telegram Bot.
    """

    def __init__(self, config: dict, openai: OpenAIHelper, db: DB, rates: dict):
        """
        Initializes the bot with the given configuration and GPT bot object.
        :param config: A dictionary containing the bot configuration
        :param openai: OpenAIHelper object
        :param db: DB object
        :param rates: Rates object
        """
        self.config = config
        self.openai = openai
        self.db = db
        self.rates = rates
        self.replicate = Replicate(api_key=config["replicate_token"])
        bot_language = self.config["bot_language"]
        with open("presets.json", "r") as f:
            self.presets = json.load(f)
        self.commands = [
            BotCommand(
                command="help",
                description=localized_text("help_description", bot_language),
            ),
            BotCommand(
                command="support",
                description=localized_text("support_description", bot_language),
            ),
            BotCommand(
                command="model",
                description="🤖 Сменить версию модели (только для тарифа GPT-4)",
            ),
            BotCommand(
                command="sdxl",
                description="🖼 Создать изображением со Stable Diffusion XL",
            ),
            BotCommand(
                command="sticker",
                description="😂 Создавайте стикеры с помощью AI",
            ),
            BotCommand(
                command="bg",
                description="🤪 Удалить фон с изображения"
            ),
            BotCommand(
                command="reset",
                description=localized_text("reset_description", bot_language),
            ),
            BotCommand(
                command="stats",
                description=localized_text("stats_description", bot_language),
            ),
            BotCommand(command="assistant", description="🤖 Сменить ассистента"),
            BotCommand(
                command="resend",
                description=localized_text("resend_description", bot_language),
            ),
            BotCommand(
                command="pay",
                description="💰 Обновить тариф",
            ),
        ]
        # If imaging is enabled, add the "image" command to the list
        if self.config.get("enable_image_generation", False):
            self.commands.append(
                BotCommand(
                    command="image",
                    description=localized_text("image_description", bot_language),
                )
            )

        if self.config.get("enable_tts_generation", False):
            self.commands.append(
                BotCommand(
                    command="voice",
                    description=localized_text("tts_description", bot_language),
                )
            )

        self.group_commands = [
                                  BotCommand(
                                      command="chat",
                                      description=localized_text("chat_description", bot_language),
                                  )
                              ] + self.commands
        self.disallowed_message = localized_text("disallowed", bot_language)
        self.budget_limit_message = localized_text("budget_limit", bot_language)
        self.usage = {}
        self.last_message = {}
        self.inline_queries_cache = {}

    async def start(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Shows the start message.
        """
        commands = self.group_commands if is_group_chat(update) else self.commands
        if not self.db.is_user_exists(update.message.chat_id):
            # Send the welcome message
            await update.message.reply_text(
                f"""Привет, {update.message.from_user.first_name}!\n\nВам активирован 
тариф “Базовый” на 3 дня, чтобы вы протестировали все функции ИИ-ассистента “Нейроскрайб”""",
                reply_markup=InlineKeyboardMarkup(
                    [
                        [
                            InlineKeyboardButton(
                                text="🚀 Отлично, продолжим!", callback_data="start_1"
                            )
                        ]
                    ]
                ),
            )
            username = update.message.from_user.username
            chat_id = update.message.chat_id
            if username:
                text = "Новый пользователь: @" + username
            else:
                text = "Новый пользователь с chat_id: " + str(chat_id)
            try:
                await update.get_bot().send_message(
                    chat_id=self.config["admin_group_id"], text=text
                )
            except telegram.error.BadRequest as e:
                print(e)
            self.db.create_user(
                username=update.message.from_user.username,
                chat_id=update.message.chat_id,
                gpt35_rate=self.rates["base"]["gpt35_rate"],
                gpt4_rate=self.rates["base"]["gpt4_rate"],
                dalle_rate=self.rates["base"]["dalle_rate"],
                whisper_rate=self.rates["base"]["whisper_rate"],
                tts_rate=self.rates["base"]["tts_rate"],
                rate_end_date=datetime.now()
                              + timedelta(days=3),  # Три дня на тестирование
                rate_type="base",
                is_free=True,
            )

        else:
            await self.help(update, _)

    async def help(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Shows the help menu.
        """
        commands = self.group_commands if is_group_chat(update) else self.commands
        commands_description = [
            f"/{command.command} - {command.description}" for command in commands
        ]
        if is_admin(self.config, update.message.from_user.id):
            commands_description.append(
                f"/admin - {localized_text('admin_description', self.config['bot_language'])}"
            )
        help_text = (
                f"""{update.message.from_user.first_name}, я твой ИИ-ассистент “Нейроскрайб”

<b>Инструкции: </b>
👉🏻 Как правильно пользоваться ботом и получить лучшие результаты
https://telegra.ph/Kak-pravilno-polzovatsya-II--nejroskrajb-02-23 

⚡️ 110 задач, которые вы можете делегировать боту. Книга-Гайд
https://neuroscribe.ru/110tasks 

⚡️Список промтов и запросов для создания контента
https://telegra.ph/Spisok-promtov-i-zaprosov-dlya-II--nejroskrajb-02-23 


<b>Вот что я умею: </b>
"""
                + "\n".join(commands_description)
                + "\n\nИИ-ассистенту можно отправлять голосовые сообщения!\n\nХотите пройти обучение?"
        )
        await update.message.reply_text(
            help_text,
            disable_web_page_preview=True,
            reply_markup=InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            f"Пройти обучение",
                            callback_data=f"start_2",
                        ),
                    ]
                ]
            ),
            parse_mode=constants.ParseMode.HTML,
        )

    async def admin(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Shows the admin menu.
        """
        if not is_admin(self.config, update.message.from_user.id):
            await update.message.reply_text(
                localized_text("admin_disallowed", self.config["bot_language"])
            )
            return

        admin_commands = [
            BotCommand(
                command="dump",
                description=localized_text(
                    "dump_description", self.config["bot_language"]
                ),
            ),
            BotCommand(
                command="mail",
                description=localized_text(
                    "mail_description", self.config["bot_language"]
                ),
            ),
            BotCommand(
                command="keys",
                description="Войти в панель управления ключами от OpenAI API",
            ),
            BotCommand(
                command="change_rate",
                description=localized_text(
                    "change_rate_description", self.config["bot_language"]
                ),
            ),
        ]
        admin_commands_description = [
            f"/{command.command} - {command.description}" for command in admin_commands
        ]
        admin_text = "\n\n".join(admin_commands_description)
        await update.message.reply_text(admin_text, disable_web_page_preview=True)

    async def dump(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Dumps the database to excel file.
        """
        if not is_admin(self.config, update.message.from_user.id):
            await update.message.reply_text(
                localized_text("admin_disallowed", self.config["bot_language"])
            )
            return

        logging.info(
            f"User {update.message.from_user.name} (id: {update.message.from_user.id}) "
            f"requested a database dump"
        )

        await update.message.reply_text(
            "Выполняю выгрузку базы данных, пожалуйста, подождите..."
        )

        users = self.db.get_all_users()
        user_data = [user.__dict__ for user in users]
        for user in user_data:
            user.pop("_sa_instance_state", None)  # Remove SQLAlchemy-specific attribute
        file_name = f"users_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.xlsx"
        df = pd.DataFrame(user_data)
        df.to_excel(file_name)

        await update.message.reply_document(
            document=open(file_name, "rb"), caption="Готово"
        )

        os.remove(file_name)

    async def keys(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        # Check if the user is an admin
        if not is_admin(self.config, update.message.from_user.id):
            await update.message.reply_text(
                localized_text("admin_disallowed", self.config["bot_language"])
            )
            return
        admin_commands = [
            BotCommand(
                command="keys_get", description="Получить список текущих ключей"
            ),
            BotCommand(
                command="keys_delete",
                description="Удалить ключ (/keys_delete <номер ключа>)",
            ),
            BotCommand(
                command="keys_add", description="Добавить ключ (/keys_add <API ключ>)"
            ),
        ]
        admin_commands_description = [
            f"/{command.command} - {command.description}" for command in admin_commands
        ]
        admin_text = "\n".join(admin_commands_description)
        await update.message.reply_text(admin_text, disable_web_page_preview=True)

    async def keys_get(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        # Check if the user is an admin
        if not is_admin(self.config, update.message.from_user.id):
            await update.message.reply_text(
                localized_text("admin_disallowed", self.config["bot_language"])
            )
            return
        keys = self.db.get_all_keys()
        keys_text = "🔑 Текущие ключи:\n\n"
        for key in keys:
            keys_text += f"{key.id}. <code>{mask_api_key(key.api_key)}</code>\n"
        await update.message.reply_text(keys_text, parse_mode=constants.ParseMode.HTML)

    async def keys_delete(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        if not is_admin(self.config, update.message.from_user.id):
            await update.message.reply_text(
                localized_text("admin_disallowed", self.config["bot_language"])
            )
            return
        key_id = (
            update.message.text.split()[1]
            if len(update.message.text.split()) > 1
            else None
        )
        if key_id is None:
            await update.message.reply_text(
                "Пожалуйста, укажите ID ключа для удаления."
            )
            return
        key = self.db.get_key_by_id(key_id)
        if key is None:
            await update.message.reply_text(f"Ключ с ID {key_id} не найден.")
            return
        self.db.delete_key(key_id)
        await update.message.reply_text(f"Ключ с ID {key_id} успешно удален.")

    async def keys_add(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        if not is_admin(self.config, update.message.from_user.id):
            await update.message.reply_text(
                localized_text("admin_disallowed", self.config["bot_language"])
            )
            return
        api_key = (
            update.message.text.split()[1]
            if len(update.message.text.split()) > 1
            else None
        )
        if api_key is None:
            await update.message.reply_text(
                "Пожалуйста, укажите API ключ для добавления."
            )
            return
        url = "https://api.openai.com/v1/models"
        headers = {"Authorization": f"Bearer {api_key}"}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    key = self.db.add_key(api_key)
                    await update.message.reply_text(f"Ключ успешно добавлен.")
                elif response.status == 401:
                    await update.message.reply_text(
                        "Неверный API ключ. Пожалуйста, проверьте правильность ключа."
                    )
                else:
                    error_message = await response.text()
                    await update.message.reply_text(
                        f"Ошибка при проверке API ключа: {error_message}"
                    )

    async def mail(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Sends a broadcast message to a group of users.
        """
        # Check if the user is an admin
        if not is_admin(self.config, update.message.from_user.id):
            await update.message.reply_text(
                localized_text("admin_disallowed", self.config["bot_language"])
            )
            return

        # Extract command arguments
        args = (
            update.message.caption.split()
            if update.message.caption
            else update.message.text.split()
        )
        if len(args) < 3:
            await update.message.reply_text("Используйте: /mail <group> <message>")
            return

        group = args[1]
        message = " ".join(args[2:])

        # Retrieve the list of users based on the group
        if group == "all":
            users = self.db.get_all_users()
        elif group == "free":
            users = self.db.get_users_by_free_or_payed(True)
        elif group == "payed":
            users = self.db.get_users_by_free_or_payed(False)
        else:
            await update.message.reply_text(
                f"Недопустимая группа '{group}'. Разрешенные группы: all, free, payed"
            )
            return
        # Send the message to each user
        for user in users:
            user = user.__dict__
            try:
                if update.message.document:
                    await context.bot.send_document(
                        chat_id=user["chat_id"],
                        document=update.message.document.file_id,
                        caption=message,
                    )
                elif update.message.animation:
                    await context.bot.send_animation(
                        chat_id=user["chat_id"],
                        animation=update.message.animation.file_id,
                        caption=message,
                    )
                elif update.message.video:
                    await context.bot.send_video(
                        chat_id=user["chat_id"],
                        video=update.message.video.file_id,
                        caption=message,
                    )
                elif update.message.photo:
                    await context.bot.send_photo(
                        chat_id=user["chat_id"],
                        photo=update.message.photo[-1].file_id,
                        caption=message,
                    )
                elif update.message.audio:
                    await context.bot.send_audio(
                        chat_id=user["chat_id"],
                        audio=update.message.audio.file_id,
                        caption=message,
                    )
                else:
                    await context.bot.send_message(
                        chat_id=user["chat_id"], text=message
                    )
            except Exception as e:
                await update.message.reply_text(
                    f"Не удалось отправить сообщение пользователю {user['chat_id']}"
                )
                logging.exception(e)

        await update.message.reply_text("Сообщения успешно отправлены")

    async def change_rate(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Changes the rate for a user. The first argument is user_id (integer) or username (string),
        the second is the rate number from four options (1, 2, 3, 4). Example: /change_rate username 2
        """
        # Check if the user is an admin
        if not is_admin(self.config, update.message.from_user.id):
            await update.message.reply_text(
                localized_text("admin_disallowed", self.config["bot_language"])
            )
            return

        args = update.message.text.split()
        if len(args) != 3:
            await update.message.reply_text(
                "Используйте: /change_rate <chat_id или username> <номер тарифа>"
            )
            return

        user_identifier, rate_number = args[1], args[2]

        if not rate_number.isdigit() or int(rate_number) not in [1, 2, 3, 4]:
            await update.message.reply_text(
                "Неверный номер тарифа. Он должен быть 1, 2, 3, или 4."
            )
            return

        rate_type = self.rates[list(self.rates.keys())[int(rate_number) - 1]]

        if user_identifier.isdigit():
            user = self.db.get_user(chat_id=int(user_identifier))
        else:
            user = self.db.get_user(username=user_identifier)

        if user is None:
            await update.message.reply_text("Пользователь не найден.")
            return

        user = user.__dict__

        self.db.update_user_field(user["chat_id"], "gpt4_rate", rate_type["gpt4_rate"])
        self.db.update_user_field(
            user["chat_id"], "gpt35_rate", rate_type["gpt35_rate"]
        )
        self.db.update_user_field(
            user["chat_id"], "dalle_rate", rate_type["dalle_rate"]
        )
        self.db.update_user_field(
            user["chat_id"], "whisper_rate", rate_type["whisper_rate"]
        )
        self.db.update_user_field(user["chat_id"], "tts_rate", rate_type["tts_rate"])
        self.db.update_user_field(
            user["chat_id"], "rate_end_date", datetime.now() + timedelta(days=30)
        )
        self.db.update_user_field(
            user["chat_id"], "rate_type", list(self.rates.keys())[int(rate_number) - 1]
        )
        self.db.update_user_field(user["chat_id"], "is_free", False)

        await update.message.reply_text(
            f"Тариф пользователя {user['chat_id']} изменен на {rate_type['name']}"
        )

    async def stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Returns token usage statistics for current day and month.
        """
        if not await is_allowed(self.config, update, context):
            logging.warning(
                f"User {update.message.from_user.name} (id: {update.message.from_user.id}) "
                f"is not allowed to request their usage statistics"
            )
            await self.send_disallowed_message(update, context)
            return

        logging.info(
            f"User {update.message.from_user.name} (id: {update.message.from_user.id}) "
            f"requested their usage statistics"
        )

        user_id = update.message.from_user.id
        if user_id not in self.usage:
            self.usage[user_id] = UsageTracker(user_id, update.message.from_user.name)

        tokens_today, tokens_month = self.usage[user_id].get_current_token_usage()
        images_today, images_month = self.usage[user_id].get_current_image_count()
        (
            transcribe_minutes_today,
            transcribe_seconds_today,
            transcribe_minutes_month,
            transcribe_seconds_month,
        ) = self.usage[user_id].get_current_transcription_duration()
        vision_today, vision_month = self.usage[user_id].get_current_vision_tokens()
        characters_today, characters_month = self.usage[user_id].get_current_tts_usage()

        chat_id = update.effective_chat.id
        chat_messages, chat_token_length = self.openai.get_conversation_stats(chat_id)
        bot_language = self.config["bot_language"]

        text_current_conversation = (
            f"<b>Текущий разговор</b>:\n"
            f"Сообщений: {chat_messages}\n"
            f"Токенов: {chat_token_length}\n"
            f"----------------------------\n"
        )

        # Check if image generation is enabled and, if so, generate the image statistics for today
        text_today_images = ""
        if self.config.get("enable_image_generation", False):
            text_today_images = f"Изображений: {images_today}\n"

        text_today_vision = ""
        if self.config.get("enable_vision", False):
            text_today_vision = vision_today

        text_today_tts = ""
        if self.config.get("enable_tts_generation", False):
            text_today_tts = f"Символов озвучено: {characters_today}\n"

        text_today = (
            f"<b>Использовано СЕГОДНЯ:</b>\n"
            f"Токенов: {tokens_today + text_today_vision}\n"
            f"{text_today_images}"
            f"{text_today_tts}"
            f"Речь в текст: {transcribe_minutes_today} {localized_text('stats_transcribe', bot_language)[0]} "
            f"{transcribe_seconds_today} {localized_text('stats_transcribe', bot_language)[1]}\n"
            f"----------------------------\n"
        )

        text_month_images = ""
        if self.config.get("enable_image_generation", False):
            text_month_images = f"Изображений: {images_month}\n"

        text_month_tts = ""
        if self.config.get("enable_tts_generation", False):
            text_month_tts = f"Символов озвучено: {characters_month}\n"

        # Check if image generation is enabled and, if so, generate the image statistics for the month
        text_month = (
            f"<b>Использовано В ЭТОМ МЕСЯЦЕ:</b>\n"
            f"Токенов: {tokens_month + vision_month}\n"
            f"{text_month_images}"  # Include the image statistics for the month if applicable
            f"{text_month_tts}"
            f"Речь в текст: {transcribe_minutes_month} {localized_text('stats_transcribe', bot_language)[0]} "
            f"{transcribe_seconds_month} {localized_text('stats_transcribe', bot_language)[1]}\n"
            f"----------------------------\n"
        )

        # Выводим остатки по тарифу и дату окончания его
        user = self.db.get_user(chat_id=update.message.from_user.id)
        text_budget = (
            f"<b>📊Вот ваша статистика, {update.message.from_user.first_name}</b>\n\n"
            f"<b>Ваш тариф: {self.rates[user.rate_type]['name']}</b>\n\n"
        )
        if self.rates[user.rate_type]["gpt4_rate"]:
            text_budget += f"<b>Токенов GPT-4 осталось:</b> {user.gpt4_rate} из {self.rates[user.rate_type]['gpt4_rate']}\n"
        text_budget += (
            f"<b>Токенов GPT-3.5 осталось:</b> {user.gpt35_rate} из {self.rates[user.rate_type]['gpt35_rate']}\n"
            f"<b>Изображений осталось:</b> {user.dalle_rate} из {self.rates[user.rate_type]['dalle_rate']}\n"
            f"<b>Речь в текст осталось:</b> {user.whisper_rate} из {self.rates[user.rate_type]['whisper_rate']}\n"
            f"<b>Озвучка осталось:</b> {user.tts_rate} из {self.rates[user.rate_type]['tts_rate']}\n"
            f"<b>Тариф действителен до:</b> {user.rate_end_date}\n"
            f"----------------------------\n"
        )
        # No longer works as of July 21st 2023, as OpenAI has removed the billing API
        # add OpenAI account information for admin request
        # if is_admin(self.config, user_id):
        #     text_budget += (
        #         f"{localized_text('stats_openai', bot_language)}"
        #         f"{self.openai.get_billing_current_month():.2f}"
        #     )

        usage_text = text_budget + text_current_conversation + text_today + text_month
        await update.message.reply_text(
            usage_text,
            parse_mode=constants.ParseMode.HTML,
            reply_markup=InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            "🔄 Обновить тариф",
                            callback_data="change_rate",
                        )
                    ]
                ]
            ),
        )

    async def resend(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Resend the last request
        """
        if not await is_allowed(self.config, update, context):
            logging.warning(
                f"User {update.message.from_user.name}  (id: {update.message.from_user.id})"
                f" is not allowed to resend the message"
            )
            await self.send_disallowed_message(update, context)
            return

        chat_id = update.effective_chat.id
        if chat_id not in self.last_message:
            logging.warning(
                f"User {update.message.from_user.name} (id: {update.message.from_user.id})"
                f" does not have anything to resend"
            )
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                text=localized_text("resend_failed", self.config["bot_language"]),
            )
            return

        # Update message text, clear self.last_message and send the request to prompt
        logging.info(
            f"Resending the last prompt from user: {update.message.from_user.name} "
            f"(id: {update.message.from_user.id})"
        )
        with update.message._unfrozen() as message:
            message.text = self.last_message.pop(chat_id)

        await self.prompt(update=update, context=context)

    async def reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Resets the conversation.
        """
        if not await is_allowed(self.config, update, context):
            logging.warning(
                f"User {update.message.from_user.name} (id: {update.message.from_user.id}) "
                f"is not allowed to reset the conversation"
            )
            await self.send_disallowed_message(update, context)
            return

        logging.info(
            f"Resetting the conversation for user {update.message.from_user.name} "
            f"(id: {update.message.from_user.id})..."
        )

        chat_id = update.effective_chat.id
        reset_content = message_text(update.message)
        self.openai.reset_chat_history(chat_id=chat_id, content=reset_content)
        user = self.db.get_user(chat_id=chat_id)
        preset = self.presets[user.default_preset]
        await update.effective_message.reply_text(
            message_thread_id=get_thread_id(update),
            parse_mode=constants.ParseMode.HTML,
            text=f"""Готово, {update.message.from_user.first_name}\nКонтекст очищен, и теперь можно начинать общение с нуля.\n\n{preset['welcome_message']}""",
        )

    async def check_rate_limit(
            self, update: Update, chat_id: int, rate_type: str = None
    ):
        user = self.db.get_user(chat_id=chat_id)
        okay = True
        if user.rate_end_date < datetime.now().date():
            await update.effective_message.reply_text(
                text=f"😢 К сожалению, {update.message.from_user.first_name}, ваша подписка закончилась..\n\n\nОбновите свой тариф, чтобы продолжить общение с нейроскрайбом",
                reply_markup=InlineKeyboardMarkup(
                    [
                        [
                            InlineKeyboardButton(
                                f"🔄 Обновить тариф",
                                callback_data="change_rate",
                            )
                        ]
                    ]
                ),
            )
            okay = False
        elif not user.__dict__.get(rate_type):
            text = f"😢 К сожалению, {update.message.from_user.first_name}, ваши токены закончились..\n\nОбновите свой тариф, чтобы продолжить общение с нейроскрайбом"
            await update.effective_message.reply_text(
                text=text,
                reply_markup=InlineKeyboardMarkup(
                    [
                        [
                            InlineKeyboardButton(
                                f"🔄 Обновить тариф",
                                callback_data="change_rate",
                            )
                        ]
                    ]
                ),
            )
            okay = False
        elif user.__dict__.get(rate_type) <= 0:
            text = None
            if rate_type.startswith("gpt"):
                text = "ваши токены закончились."
            if rate_type == "dalle_rate":
                text = "ваши изображения закончились."
            if rate_type == "whisper_rate":
                text = "ваше время для расшифровки текста закончилось"
            if rate_type == "tts_rate":
                text = "ваши символы для озвучки текста закончились"
            await update.effective_message.reply_text(
                text=f"😢 К сожалению, {update.message.from_user.first_name}, {text}.\n\nОбновите свой тариф, чтобы продолжить общение с нейроскрайбом",
                reply_markup=InlineKeyboardMarkup(
                    [
                        [
                            InlineKeyboardButton(
                                localized_text(
                                    "change_rate", self.config["bot_language"]
                                ),
                                callback_data="change_rate",
                            )
                        ]
                    ]
                ),
            )
            okay = False
        return user, okay

    async def sticker(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        image_query = message_text(update.message)
        if image_query == "":
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                text="""🎨 Для генерации стикера, пожалуйста, <b>введите запрос используя команду /sticker

Например:</b> /sticker a cat""",
                parse_mode=constants.ParseMode.HTML,
            )
            return
        image_url = self.replicate.run(
            "6443cc831f51eb01333f50b757157411d7cadb6215144cc721e3688b70004ad0",
            {
                "steps": 20,
                "width": 1024,
                "height": 1024,
                "prompt": image_query,
                "upscale": False,
                "upscale_steps": 10,
                "negative_prompt": "",
            },
        )
        # send image to user
        await update.effective_message.reply_photo(
            reply_to_message_id=get_reply_to_message_id(self.config, update),
            photo=image_url[1],
        )
        return

    async def bg(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not update.message.photo:
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                text="""🖼️ Для удаления фона, пожалуйста, <b>отправьте фотографию с командой /bg в подписи</b>""",
                parse_mode=constants.ParseMode.HTML,
            )
            return

        photo = update.message.photo[-1]
        photo_file = await context.bot.get_file(photo.file_id)
        photo_url = photo_file.file_path

        image_url = self.replicate.run(
            "f91971acb059a5b9e29bf3ad451c9bc4dc807a719427037a6623302ddc598e35",
            {
                "image": photo_url
            },
        )[0]
        # send image to user
        await update.effective_message.reply_photo(
            reply_to_message_id=get_reply_to_message_id(self.config, update),
            photo=image_url,
        )
        return

    async def sdxl(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        image_query = message_text(update.message)
        if image_query == "":
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                text="""🎨 Для генерации изображений, пожалуйста, <b>введите запрос используя команду /sdxl

    Например:</b> /sdxl a cat""",
                parse_mode=constants.ParseMode.HTML,
            )
            return
        image_url = self.replicate.run(
            "39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
            {
                "width": 768,
                "height": 768,
                "prompt": image_query,
                "refine": "expert_ensemble_refiner",
                "scheduler": "K_EULER",
                "lora_scale": 0.6,
                "num_outputs": 1,
                "guidance_scale": 7.5,
                "apply_watermark": False,
                "high_noise_frac": 0.8,
                "negative_prompt": "",
                "prompt_strength": 0.8,
                "num_inference_steps": 25,
            },
        )[0]
        # send image to user
        await update.effective_message.reply_photo(
            reply_to_message_id=get_reply_to_message_id(self.config, update),
            photo=image_url,
        )
        return

    async def image(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Generates an image for the given prompt using DALL·E APIs
        """
        if not self.config[
            "enable_image_generation"
        ] or not await self.check_allowed_and_within_budget(update, context):
            return

        # Проверяем действительность тарифа
        user, okay = await self.check_rate_limit(
            update, update.effective_chat.id, "dalle_rate"
        )

        if not okay:
            return

        image_query = message_text(update.message)
        if image_query == "":
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                text="""🎨 Для генерации изображений, пожалуйста, <b>введите запрос используя команду /image

Например:</b> /image кошка""",
                parse_mode=constants.ParseMode.HTML,
            )
            return

        logging.info(
            f"New image generation request received from user {update.message.from_user.name} "
            f"(id: {update.message.from_user.id})"
        )

        async def _generate():
            try:
                image_url, image_size = await self.openai.generate_image(
                    prompt=image_query
                )
                if self.config["image_receive_mode"] == "photo":
                    await update.effective_message.reply_photo(
                        reply_to_message_id=get_reply_to_message_id(
                            self.config, update
                        ),
                        photo=image_url,
                    )
                elif self.config["image_receive_mode"] == "document":
                    await update.effective_message.reply_document(
                        reply_to_message_id=get_reply_to_message_id(
                            self.config, update
                        ),
                        document=image_url,
                    )
                else:
                    raise Exception(
                        f"env variable IMAGE_RECEIVE_MODE has invalid value {self.config['image_receive_mode']}"
                    )
                # add image request to users usage tracker
                user_id = update.message.from_user.id
                self.usage[user_id].add_image_request(
                    image_size, self.config["image_prices"]
                )
                # списываем с баланса
                self.db.update_user_field(
                    chat_id=update.message.from_user.id,
                    field_name="dalle_rate",
                    new_value=user.dalle_rate - 1,
                )
                # add guest chat request to guest usage tracker
                if (
                        str(user_id) not in self.config["allowed_user_ids"].split(",")
                        and "guests" in self.usage
                ):
                    self.usage["guests"].add_image_request(
                        image_size, self.config["image_prices"]
                    )

            except Exception as e:
                logging.exception(e)
                await update.effective_message.reply_text(
                    message_thread_id=get_thread_id(update),
                    reply_to_message_id=get_reply_to_message_id(self.config, update),
                    text=f"{localized_text('image_fail', self.config['bot_language'])}: {str(e)}",
                    parse_mode=constants.ParseMode.MARKDOWN,
                )

        await wrap_with_indicator(
            update, context, _generate, constants.ChatAction.UPLOAD_PHOTO
        )

    async def tts(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Generates an speech for the given input using TTS APIs
        """
        if not self.config[
            "enable_tts_generation"
        ] or not await self.check_allowed_and_within_budget(update, context):
            return

        # Проверяем действительность тарифа
        user, okay = await self.check_rate_limit(
            update, update.effective_chat.id, "tts_rate"
        )

        if not okay:
            return

        tts_query = message_text(update.message)
        if tts_query == "":
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                text="""🔈 Для озвучки текста, пожалуйста, <b>введите текст используя команду /voice

Например:</b> /voice Всем привет, меня зовут нейроскрайб! Я ИИ-ассистент для экспертов и создателей контента""",
                parse_mode=constants.ParseMode.HTML,
            )
            return

        logging.info(
            f"New speech generation request received from user {update.message.from_user.name} "
            f"(id: {update.message.from_user.id})"
        )

        async def _generate():
            try:
                speech_file, text_length = await self.openai.generate_speech(
                    text=tts_query
                )

                await update.effective_message.reply_voice(
                    reply_to_message_id=get_reply_to_message_id(self.config, update),
                    voice=speech_file,
                )
                speech_file.close()
                # add image request to users usage tracker
                user_id = update.message.from_user.id
                self.usage[user_id].add_tts_request(
                    text_length, self.config["tts_model"], self.config["tts_prices"]
                )
                self.db.update_user_field(
                    chat_id=update.message.from_user.id,
                    field_name="tts_rate",
                    new_value=user.tts_rate - text_length,
                )
                # add guest chat request to guest usage tracker
                if (
                        str(user_id) not in self.config["allowed_user_ids"].split(",")
                        and "guests" in self.usage
                ):
                    self.usage["guests"].add_tts_request(
                        text_length, self.config["tts_model"], self.config["tts_prices"]
                    )

            except Exception as e:
                logging.exception(e)
                await update.effective_message.reply_text(
                    message_thread_id=get_thread_id(update),
                    reply_to_message_id=get_reply_to_message_id(self.config, update),
                    text=f"{localized_text('tts_fail', self.config['bot_language'])}: {str(e)}",
                    parse_mode=constants.ParseMode.MARKDOWN,
                )

        await wrap_with_indicator(
            update, context, _generate, constants.ChatAction.UPLOAD_VOICE
        )

    async def transcribe(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Transcribe audio messages.
        """
        if not self.config[
            "enable_transcription"
        ] or not await self.check_allowed_and_within_budget(update, context):
            return

        if update.message.caption and update.message.caption.startswith("/mail"):
            await self.mail(update, context)
            return

        # Проверяем действительность тарифа
        user, okay = await self.check_rate_limit(
            update, update.effective_chat.id, "whisper_rate"
        )

        if not okay:
            return

        # Проверяем действительность тарифа
        if user.rate_type == "gpt-4":
            user, okay = await self.check_rate_limit(
                update, update.effective_chat.id, "gpt4_rate"
            )

            if not okay:
                return
        else:
            if user.rate_type == "gpt-4":
                user, okay = await self.check_rate_limit(
                    update, update.effective_chat.id, "gpt35_rate"
                )

                if not okay:
                    return

        if is_group_chat(update) and self.config["ignore_group_transcriptions"]:
            logging.info(f"Transcription coming from group chat, ignoring...")
            return

        chat_id = update.effective_chat.id
        filename = update.message.effective_attachment.file_unique_id

        async def _execute():
            filename_mp3 = f"{filename}.mp3"
            bot_language = self.config["bot_language"]
            try:
                media_file = await context.bot.get_file(
                    update.message.effective_attachment.file_id
                )
                await media_file.download_to_drive(filename)
            except Exception as e:
                logging.exception(e)
                await update.effective_message.reply_text(
                    message_thread_id=get_thread_id(update),
                    reply_to_message_id=get_reply_to_message_id(self.config, update),
                    text=(
                        f"{localized_text('media_download_fail', bot_language)[0]}: "
                        f"{str(e)}. {localized_text('media_download_fail', bot_language)[1]}"
                    ),
                    parse_mode=constants.ParseMode.MARKDOWN,
                )
                return

            try:
                audio_track = AudioSegment.from_file(filename)
                audio_track.export(filename_mp3, format="mp3")
                logging.info(
                    f"New transcribe request received from user {update.message.from_user.name} "
                    f"(id: {update.message.from_user.id})"
                )

            except Exception as e:
                logging.exception(e)
                await update.effective_message.reply_text(
                    message_thread_id=get_thread_id(update),
                    reply_to_message_id=get_reply_to_message_id(self.config, update),
                    text=localized_text("media_type_fail", bot_language),
                )
                if os.path.exists(filename):
                    os.remove(filename)
                return

            user_id = update.message.from_user.id
            if user_id not in self.usage:
                self.usage[user_id] = UsageTracker(
                    user_id, update.message.from_user.name
                )

            try:
                transcript = await self.openai.transcribe(filename_mp3)

                transcription_price = self.config["transcription_price"]
                self.usage[user_id].add_transcription_seconds(
                    audio_track.duration_seconds, transcription_price
                )
                self.db.update_user_field(
                    chat_id=update.message.from_user.id,
                    field_name="whisper_rate",
                    new_value=user.whisper_rate - 1,
                )

                allowed_user_ids = self.config["allowed_user_ids"].split(",")
                if str(user_id) not in allowed_user_ids and "guests" in self.usage:
                    self.usage["guests"].add_transcription_seconds(
                        audio_track.duration_seconds, transcription_price
                    )

                # check if transcript starts with any of the prefixes
                response_to_transcription = any(
                    transcript.lower().startswith(prefix.lower()) if prefix else False
                    for prefix in self.config["voice_reply_prompts"]
                )

                if (
                        self.config["voice_reply_transcript"]
                        and not response_to_transcription
                ):
                    # Split into chunks of 4096 characters (Telegram's message limit)
                    transcript_output = f"_{localized_text('transcript', bot_language)}:_\n\"{transcript}\""
                    chunks = split_into_chunks(transcript_output)

                    for index, transcript_chunk in enumerate(chunks):
                        await update.effective_message.reply_text(
                            message_thread_id=get_thread_id(update),
                            reply_to_message_id=get_reply_to_message_id(
                                self.config, update
                            )
                            if index == 0
                            else None,
                            text=transcript_chunk,
                            parse_mode=constants.ParseMode.MARKDOWN,
                        )
                else:
                    # Get the response of the transcript
                    response, total_tokens = await self.openai.get_chat_response(
                        chat_id=chat_id, query=transcript
                    )

                    self.usage[user_id].add_chat_tokens(
                        total_tokens, self.config["token_price"]
                    )
                    if str(user_id) not in allowed_user_ids and "guests" in self.usage:
                        self.usage["guests"].add_chat_tokens(
                            total_tokens, self.config["token_price"]
                        )

                    if user.rate_type == "gpt-4":
                        self.db.update_user_field(
                            chat_id=update.message.from_user.id,
                            field_name="gpt4_rate",
                            new_value=user.gpt4_rate - total_tokens,
                        )
                    else:
                        self.db.update_user_field(
                            chat_id=update.message.from_user.id,
                            field_name="gpt35_rate",
                            new_value=user.gpt35_rate - total_tokens,
                        )

                    # Split into chunks of 4096 characters (Telegram's message limit)
                    transcript_output = (
                        f"_{localized_text('transcript', bot_language)}:_\n\"{transcript}\"\n\n"
                        f"_{localized_text('answer', bot_language)}:_\n{response}"
                    )
                    chunks = split_into_chunks(transcript_output)

                    for index, transcript_chunk in enumerate(chunks):
                        await update.effective_message.reply_text(
                            message_thread_id=get_thread_id(update),
                            reply_to_message_id=get_reply_to_message_id(
                                self.config, update
                            )
                            if index == 0
                            else None,
                            text=transcript_chunk,
                            parse_mode=constants.ParseMode.MARKDOWN,
                        )

            except Exception as e:
                logging.exception(e)
                await update.effective_message.reply_text(
                    message_thread_id=get_thread_id(update),
                    reply_to_message_id=get_reply_to_message_id(self.config, update),
                    text=f"{localized_text('transcribe_fail', bot_language)}: {str(e)}",
                    parse_mode=constants.ParseMode.MARKDOWN,
                )
            finally:
                if os.path.exists(filename_mp3):
                    os.remove(filename_mp3)
                if os.path.exists(filename):
                    os.remove(filename)

        await wrap_with_indicator(
            update, context, _execute, constants.ChatAction.TYPING
        )

    async def vision(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Interpret image using vision model.
        """
        if not self.config[
            "enable_vision"
        ] or not await self.check_allowed_and_within_budget(update, context):
            return

        if update.message.caption and update.message.caption.startswith("/mail"):
            await self.mail(update, context)
            return

        if update.message.caption and update.message.caption.startswith("/bg"):
            await self.bg(update, context)
            return

        # Проверяем действительность тарифа
        user, okay = await self.check_rate_limit(
            update, update.effective_chat.id, "gpt4_rate"
        )

        if not okay:
            return

        chat_id = update.effective_chat.id
        prompt = update.message.caption

        if is_group_chat(update):
            if self.config["ignore_group_vision"]:
                logging.info(f"Vision coming from group chat, ignoring...")
                return
            else:
                trigger_keyword = self.config["group_trigger_keyword"]
                if (prompt is None and trigger_keyword != "") or (
                        prompt is not None
                        and not prompt.lower().startswith(trigger_keyword.lower())
                ):
                    logging.info(
                        f"Vision coming from group chat with wrong keyword, ignoring..."
                    )
                    return

        image = update.message.effective_attachment[-1]

        async def _execute():
            bot_language = self.config["bot_language"]
            try:
                media_file = await context.bot.get_file(image.file_id)
                temp_file = io.BytesIO(await media_file.download_as_bytearray())
            except Exception as e:
                logging.exception(e)
                await update.effective_message.reply_text(
                    message_thread_id=get_thread_id(update),
                    reply_to_message_id=get_reply_to_message_id(self.config, update),
                    text=(
                        f"{localized_text('media_download_fail', bot_language)[0]}: "
                        f"{str(e)}. {localized_text('media_download_fail', bot_language)[1]}"
                    ),
                    parse_mode=constants.ParseMode.MARKDOWN,
                )
                return

            # convert jpg from telegram to png as understood by openai

            temp_file_png = io.BytesIO()

            try:
                original_image = Image.open(temp_file)

                original_image.save(temp_file_png, format="PNG")
                logging.info(
                    f"New vision request received from user {update.message.from_user.name} "
                    f"(id: {update.message.from_user.id})"
                )

            except Exception as e:
                logging.exception(e)
                await update.effective_message.reply_text(
                    message_thread_id=get_thread_id(update),
                    reply_to_message_id=get_reply_to_message_id(self.config, update),
                    text=localized_text("media_type_fail", bot_language),
                )

            user_id = update.message.from_user.id
            if user_id not in self.usage:
                self.usage[user_id] = UsageTracker(
                    user_id, update.message.from_user.name
                )

            if self.config["stream"]:
                stream_response = self.openai.interpret_image_stream(
                    chat_id=chat_id, fileobj=temp_file_png, prompt=prompt
                )
                i = 0
                prev = ""
                sent_message = None
                backoff = 0
                stream_chunk = 0

                async for content, tokens in stream_response:
                    if is_direct_result(content):
                        return await handle_direct_result(self.config, update, content)

                    if len(content.strip()) == 0:
                        continue

                    stream_chunks = split_into_chunks(content)
                    if len(stream_chunks) > 1:
                        content = stream_chunks[-1]
                        if stream_chunk != len(stream_chunks) - 1:
                            stream_chunk += 1
                            try:
                                await edit_message_with_retry(
                                    context,
                                    chat_id,
                                    str(sent_message.message_id),
                                    stream_chunks[-2],
                                )
                            except:
                                pass
                            try:
                                sent_message = (
                                    await update.effective_message.reply_text(
                                        message_thread_id=get_thread_id(update),
                                        text=content if len(content) > 0 else "...",
                                    )
                                )
                            except:
                                pass
                            continue

                    cutoff = get_stream_cutoff_values(update, content)
                    cutoff += backoff

                    if i == 0:
                        try:
                            if sent_message is not None:
                                await context.bot.delete_message(
                                    chat_id=sent_message.chat_id,
                                    message_id=sent_message.message_id,
                                )
                            sent_message = await update.effective_message.reply_text(
                                message_thread_id=get_thread_id(update),
                                reply_to_message_id=get_reply_to_message_id(
                                    self.config, update
                                ),
                                text=content,
                            )
                        except:
                            continue

                    elif (
                            abs(len(content) - len(prev)) > cutoff
                            or tokens != "not_finished"
                    ):
                        prev = content

                        try:
                            use_markdown = tokens != "not_finished"
                            await edit_message_with_retry(
                                context,
                                chat_id,
                                str(sent_message.message_id),
                                text=content,
                                markdown=use_markdown,
                            )

                        except RetryAfter as e:
                            backoff += 5
                            await asyncio.sleep(e.retry_after)
                            continue

                        except TimedOut:
                            backoff += 5
                            await asyncio.sleep(0.5)
                            continue

                        except Exception:
                            backoff += 5
                            continue

                        await asyncio.sleep(0.01)

                    i += 1
                    if tokens != "not_finished":
                        total_tokens = int(tokens)

            else:
                try:
                    interpretation, total_tokens = await self.openai.interpret_image(
                        chat_id, temp_file_png, prompt=prompt
                    )

                    try:
                        await update.effective_message.reply_text(
                            message_thread_id=get_thread_id(update),
                            reply_to_message_id=get_reply_to_message_id(
                                self.config, update
                            ),
                            text=interpretation,
                            parse_mode=constants.ParseMode.MARKDOWN,
                        )
                    except BadRequest:
                        try:
                            await update.effective_message.reply_text(
                                message_thread_id=get_thread_id(update),
                                reply_to_message_id=get_reply_to_message_id(
                                    self.config, update
                                ),
                                text=interpretation,
                            )
                        except Exception as e:
                            logging.exception(e)
                            await update.effective_message.reply_text(
                                message_thread_id=get_thread_id(update),
                                reply_to_message_id=get_reply_to_message_id(
                                    self.config, update
                                ),
                                text=f"{localized_text('vision_fail', bot_language)}: {str(e)}",
                                parse_mode=constants.ParseMode.MARKDOWN,
                            )
                except Exception as e:
                    logging.exception(e)
                    await update.effective_message.reply_text(
                        message_thread_id=get_thread_id(update),
                        reply_to_message_id=get_reply_to_message_id(
                            self.config, update
                        ),
                        text=f"{localized_text('vision_fail', bot_language)}: {str(e)}",
                        parse_mode=constants.ParseMode.MARKDOWN,
                    )
            vision_token_price = self.config["vision_token_price"]
            self.usage[user_id].add_vision_tokens(total_tokens, vision_token_price)

            self.db.update_user_field(
                chat_id=update.message.from_user.id,
                field_name="gpt4_rate",
                new_value=user.gpt4_rate - total_tokens - vision_token_price,
            )

            allowed_user_ids = self.config["allowed_user_ids"].split(",")
            if str(user_id) not in allowed_user_ids and "guests" in self.usage:
                self.usage["guests"].add_vision_tokens(total_tokens, vision_token_price)

        await wrap_with_indicator(
            update, context, _execute, constants.ChatAction.TYPING
        )

    async def support(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Shows the support message.
        """
        await update.message.reply_text(
            localized_text("support_text", self.config["bot_language"]),
            reply_markup=InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            text="Поддержка", url="https://t.me/maxnagovitsyn"
                        )
                    ]
                ]
            ),
        )

    async def model(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Change model (GPT-3.5 or GPT-4)
        """
        user = self.db.get_user(chat_id=update.effective_chat.id)
        if user.rate_type == "gpt-4":
            default_model = user.default_model
            await update.message.reply_text(
                "🤖 Выберите модель для общения:",
                reply_markup=model_keyboard(default_model),
            )
        else:
            await update.message.reply_text(
                "❌ Сменить модель можно только на тарифе GPT-4",
                reply_markup=InlineKeyboardMarkup(
                    [
                        [
                            InlineKeyboardButton(
                                "🔄 Обновить тариф",
                                callback_data="change_rate",
                            )
                        ]
                    ]
                ),
            )

    async def assistant(
            self, update: Update, _: ContextTypes.DEFAULT_TYPE, page=None, message=None
    ) -> None:
        user = self.db.get_user(chat_id=update.effective_chat.id)
        presets = self.presets
        presets_keys = list(presets.keys())

        if user.rate_type == "base":
            presets_keys = presets_keys[:5]

        page_size = 5
        total_pages = (len(presets_keys) + page_size - 1) // page_size
        current_page = page or 1

        start_index = (current_page - 1) * page_size
        end_index = min(start_index + page_size, len(presets_keys))

        keyboard = [
            [
                InlineKeyboardButton(
                    presets[preset]["name"],
                    callback_data="change_mode_" + preset,
                )
            ]
            for preset in presets_keys[start_index:end_index]
        ]

        if total_pages > 1:
            navigation_buttons = []
            if current_page > 1:
                navigation_buttons.append(
                    InlineKeyboardButton(
                        "<<", callback_data=f"assistant_page_{current_page - 1}"
                    )
                )
            if current_page < total_pages:
                navigation_buttons.append(
                    InlineKeyboardButton(
                        ">>", callback_data=f"assistant_page_{current_page + 1}"
                    )
                )
            keyboard.append(navigation_buttons)

        if message:
            await update.get_bot().editMessageText(
                chat_id=update.effective_chat.id,
                message_id=message,
                text=f"Выберите ассистента для общения (Страница {current_page} из {total_pages}):",
            )
            await update.get_bot().editMessageReplyMarkup(
                chat_id=update.effective_chat.id,
                message_id=message,
                reply_markup=InlineKeyboardMarkup(keyboard),
            )
        else:
            await update.message.reply_text(
                f"Выберите ассистента для общения (Страница {current_page} из {total_pages}):",
                reply_markup=InlineKeyboardMarkup(keyboard),
            )

    async def prompt(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        React to incoming messages and respond accordingly.
        """
        if update.edited_message or not update.message or update.message.via_bot:
            return

        user = self.db.get_user(chat_id=update.effective_chat.id)

        # Проверяем действительность тарифа
        if user.rate_type == "gpt-4":
            user, okay = await self.check_rate_limit(
                update, update.effective_chat.id, "gpt4_rate"
            )

            if not okay:
                return
        else:
            user, okay = await self.check_rate_limit(
                update, update.effective_chat.id, "gpt35_rate"
            )

            if not okay:
                return

        logging.info(
            f"New message received from user {update.message.from_user.name} (id: {update.message.from_user.id})"
        )
        chat_id = update.effective_chat.id
        user_id = update.message.from_user.id
        prompt = message_text(update.message)
        self.last_message[chat_id] = prompt

        if is_group_chat(update):
            trigger_keyword = self.config["group_trigger_keyword"]

            if prompt.lower().startswith(
                    trigger_keyword.lower()
            ) or update.message.text.lower().startswith("/chat"):
                if prompt.lower().startswith(trigger_keyword.lower()):
                    prompt = prompt[len(trigger_keyword):].strip()

                if (
                        update.message.reply_to_message
                        and update.message.reply_to_message.text
                        and update.message.reply_to_message.from_user.id != context.bot.id
                ):
                    prompt = f'"{update.message.reply_to_message.text}" {prompt}'
            else:
                if (
                        update.message.reply_to_message
                        and update.message.reply_to_message.from_user.id == context.bot.id
                ):
                    logging.info("Message is a reply to the bot, allowing...")
                else:
                    logging.warning(
                        "Message does not start with trigger keyword, ignoring..."
                    )
                    return

        try:
            total_tokens = 0

            if self.config["stream"]:
                await update.effective_message.reply_chat_action(
                    action=constants.ChatAction.TYPING,
                    message_thread_id=get_thread_id(update),
                )

                stream_response = self.openai.get_chat_response_stream(
                    chat_id=chat_id, query=prompt
                )
                i = 0
                prev = ""
                sent_message = None
                backoff = 0
                stream_chunk = 0

                async for content, tokens in stream_response:
                    if is_direct_result(content):
                        return await handle_direct_result(self.config, update, content)

                    if len(content.strip()) == 0:
                        continue

                    stream_chunks = split_into_chunks(content)
                    if len(stream_chunks) > 1:
                        content = stream_chunks[-1]
                        if stream_chunk != len(stream_chunks) - 1:
                            stream_chunk += 1
                            try:
                                await edit_message_with_retry(
                                    context,
                                    chat_id,
                                    str(sent_message.message_id),
                                    stream_chunks[-2],
                                )
                            except:
                                pass
                            try:
                                sent_message = (
                                    await update.effective_message.reply_text(
                                        message_thread_id=get_thread_id(update),
                                        text=content if len(content) > 0 else "...",
                                    )
                                )
                            except:
                                pass
                            continue

                    cutoff = get_stream_cutoff_values(update, content)
                    cutoff += backoff

                    if i == 0:
                        try:
                            if sent_message is not None:
                                await context.bot.delete_message(
                                    chat_id=sent_message.chat_id,
                                    message_id=sent_message.message_id,
                                )
                            sent_message = await update.effective_message.reply_text(
                                message_thread_id=get_thread_id(update),
                                reply_to_message_id=get_reply_to_message_id(
                                    self.config, update
                                ),
                                text=content,
                            )
                        except:
                            continue

                    elif (
                            abs(len(content) - len(prev)) > cutoff
                            or tokens != "not_finished"
                    ):
                        prev = content

                        try:
                            use_markdown = tokens != "not_finished"
                            await edit_message_with_retry(
                                context,
                                chat_id,
                                str(sent_message.message_id),
                                text=content,
                                markdown=use_markdown,
                            )

                        except RetryAfter as e:
                            backoff += 5
                            await asyncio.sleep(e.retry_after)
                            continue

                        except TimedOut:
                            backoff += 5
                            await asyncio.sleep(0.5)
                            continue

                        except Exception:
                            backoff += 5
                            continue

                        await asyncio.sleep(0.01)

                    i += 1
                    if tokens != "not_finished":
                        total_tokens = int(tokens)

            else:

                async def _reply():
                    nonlocal total_tokens
                    response, total_tokens = await self.openai.get_chat_response(
                        chat_id=chat_id, query=prompt
                    )

                    if is_direct_result(response):
                        return await handle_direct_result(self.config, update, response)

                    # Split into chunks of 4096 characters (Telegram's message limit)
                    chunks = split_into_chunks(response)

                    for index, chunk in enumerate(chunks):
                        try:
                            await update.effective_message.reply_text(
                                message_thread_id=get_thread_id(update),
                                reply_to_message_id=get_reply_to_message_id(
                                    self.config, update
                                )
                                if index == 0
                                else None,
                                text=chunk,
                                parse_mode=constants.ParseMode.MARKDOWN,
                            )
                        except Exception:
                            try:
                                await update.effective_message.reply_text(
                                    message_thread_id=get_thread_id(update),
                                    reply_to_message_id=get_reply_to_message_id(
                                        self.config, update
                                    )
                                    if index == 0
                                    else None,
                                    text=chunk,
                                )
                            except Exception as exception:
                                raise exception

                await wrap_with_indicator(
                    update, context, _reply, constants.ChatAction.TYPING
                )

            add_chat_request_to_usage_tracker(
                self.usage, self.config, user_id, total_tokens
            )

            if user.rate_type == "gpt-4":
                self.db.update_user_field(
                    chat_id=update.message.from_user.id,
                    field_name="gpt4_rate",
                    new_value=user.gpt4_rate - total_tokens,
                )
            else:
                self.db.update_user_field(
                    chat_id=update.message.from_user.id,
                    field_name="gpt35_rate",
                    new_value=user.gpt35_rate - total_tokens,
                )

        except Exception as e:
            logging.exception(e)
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                reply_to_message_id=get_reply_to_message_id(self.config, update),
                text=f"{localized_text('chat_fail', self.config['bot_language'])} {str(e)}",
                parse_mode=constants.ParseMode.MARKDOWN,
            )

    async def inline_query(
            self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """
        Handle the inline query. This is run when you type: @botusername <query>
        """
        query = update.inline_query.query
        if len(query) < 3:
            return
        if not await self.check_allowed_and_within_budget(
                update, context, is_inline=True
        ):
            return

        callback_data_suffix = "gpt:"
        result_id = str(uuid4())
        self.inline_queries_cache[result_id] = query
        callback_data = f"{callback_data_suffix}{result_id}"

        await self.send_inline_query_result(
            update, result_id, message_content=query, callback_data=callback_data
        )

    async def send_inline_query_result(
            self, update: Update, result_id, message_content, callback_data=""
    ):
        """
        Send inline query result
        """
        try:
            reply_markup = None
            bot_language = self.config["bot_language"]
            if callback_data:
                reply_markup = InlineKeyboardMarkup(
                    [
                        [
                            InlineKeyboardButton(
                                text=f'🤖 {localized_text("answer_with_chatgpt", bot_language)}',
                                callback_data=callback_data,
                            )
                        ]
                    ]
                )

            inline_query_result = InlineQueryResultArticle(
                id=result_id,
                title=localized_text("ask_chatgpt", bot_language),
                input_message_content=InputTextMessageContent(message_content),
                description=message_content,
                thumb_url="https://user-images.githubusercontent.com/11541888/223106202-7576ff11-2c8e-408d-94ea"
                          "-b02a7a32149a.png",
                reply_markup=reply_markup,
            )

            await update.inline_query.answer([inline_query_result], cache_time=0)
        except Exception as e:
            logging.error(
                f"An error occurred while generating the result card for inline query {e}"
            )

    async def pay(self, update: Update, _: ContextTypes.DEFAULT_TYPE):
        """
        Send callback query
        """
        rates_text = [
            (
                f"*{idx + 1}. {rate['name']}*\n"
                f"Токенов GPT-4: {rate['gpt4_rate'] if rate['gpt4_rate'] else 'нет'}\n"
                f"Токенов GPT-3.5: {rate['gpt35_rate']}\n"
                f"Изображений: {rate['dalle_rate']}\n"
                f"Речь в текст: {rate['whisper_rate']} минут\n"
                f"Озвучка: {rate['tts_rate']} символов\n"
                f"Стоимость: {rate['price']} рублей / месяц\n"
                f"----------------------------\n"
            )
            for idx, rate in enumerate(self.rates.values())
        ]
        await update.message.reply_text(
            text=(
                    "*Все тарифы:*\n"
                    + "".join(rates_text)
                    + "🙂Пожалуйста, выберите подходящий тариф, и нажмите на одну из кнопок, для перехода к оплате."
            ),
            parse_mode=constants.ParseMode.MARKDOWN,
            reply_markup=InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            f"{i + 1}",
                            callback_data=f"buy_rate{i}",
                        )
                        for i in range(len(self.rates))
                    ]
                ]
            ),
        )

    async def handle_callback_inline_query(
            self, update: Update, context: CallbackContext
    ):
        """
        Handle the callback query from the inline query result
        """
        callback_data = update.callback_query.data
        user_id = update.callback_query.from_user.id
        inline_message_id = update.callback_query.inline_message_id
        name = update.callback_query.from_user.name
        callback_data_suffix = "gpt:"
        query = ""
        bot_language = self.config["bot_language"]
        answer_tr = localized_text("answer", bot_language)
        loading_tr = localized_text("loading", bot_language)

        if callback_data.startswith("assistant_page_"):
            page = int(callback_data.split("assistant_page_")[-1])
            await self.assistant(
                update,
                context,
                page=page,
                message=update.callback_query.message.message_id,
            )

        if callback_data.startswith("change_model_"):
            model = callback_data.split("change_model_")[-1]
            user = self.db.get_user(chat_id=update.effective_chat.id)
            if user.default_model == model:
                return
            if user.rate_type != "gpt-4":
                await update.get_bot().send_message(
                    text="❌ Сменить модель можно только на тарифе GPT-4",
                    chat_id=update.callback_query.from_user.id,
                    reply_markup=InlineKeyboardMarkup(
                        [
                            [
                                InlineKeyboardButton(
                                    "🔄 Обновить тариф",
                                    callback_data="change_rate",
                                )
                            ]
                        ]
                    ),
                )
                return
            self.db.update_user_field(update.effective_chat.id, "default_model", model)
            await context.bot.editMessageReplyMarkup(
                message_id=update.callback_query.message.message_id,
                chat_id=update.effective_chat.id,
                reply_markup=model_keyboard(model),
            )

        if callback_data.startswith("change_mode_"):
            mode = callback_data.split("change_mode_")[-1]
            preset = self.presets[mode]
            self.db.update_user_field(update.effective_chat.id, "default_preset", mode)
            await update.get_bot().send_message(
                text=preset["welcome_message"],
                chat_id=update.callback_query.from_user.id,
                parse_mode=constants.ParseMode.HTML,
            )
            # delete old message
            await update.get_bot().delete_message(
                chat_id=update.callback_query.from_user.id,
                message_id=update.callback_query.message.message_id,
            )

        if callback_data.startswith("start_"):
            stage = callback_data.split("start_")[-1]
            if stage == "1":
                await update.get_bot().send_message(
                    text=(
                        f"""Отлично, тариф “Базовый” активирован на 3 дня
🔥 Нейроскрайб  — ИИ ассистент для экспертов и создателей контента. Основная задача этого бота на 80% улучшить вашу производительность. 

👉🏻 Как правильно пользоваться ботом и получить лучшие результаты:
https://telegra.ph/Kak-pravilno-polzovatsya-II--nejroskrajb-02-23 

⚡️ 110 задач, которые вы можете делегировать боту. Книга-Гайд
https://neuroscribe.ru/110tasks 

⚡️Список промтов и запросов для создания контента
https://telegra.ph/Spisok-promtov-i-zaprosov-dlya-II--nejroskrajb-02-23 

⚡️ Наш основной канал:
@neuroscribe 

Вы можете начать использовать бота прямо сейчас. Также, можете отправлять ему голосовые сообщения! 

{update.callback_query.from_user.first_name}, хотите пройти обучение и понять, как эффективно пользоваться ботом?
"""
                    ),
                    chat_id=update.callback_query.from_user.id,
                    parse_mode=constants.ParseMode.HTML,
                    reply_markup=InlineKeyboardMarkup(
                        [
                            [
                                InlineKeyboardButton(
                                    f"Пройти обучение",
                                    callback_data=f"start_2",
                                ),
                                InlineKeyboardButton(
                                    f"Начать общение с нейросетью",
                                    callback_data=f"start_reset",
                                ),
                            ]
                        ]
                    ),
                )
            elif stage == "reset":
                user = self.db.get_user(chat_id=update.callback_query.from_user.id)
                preset = self.presets[user.default_preset]
                await update.get_bot().send_message(
                    text=f"""Готово, {update.message.from_user.first_name}\nКонтекст очищен, и теперь можно начинать общение с нуля.\n\n{preset['welcome_message']}""",
                    chat_id=update.callback_query.from_user.id,
                    parse_mode=constants.ParseMode.HTML,
                )
            elif stage == "2":
                await update.get_bot().send_message(
                    text=(
                        f"""<b>1. Формулируйте вопрос подробно и правильно</b>

❌ Составь контент план для студии растяжки

✅ Действуй, как опытный копирайтер. Напиши контент план в виде таблицы на 2 недели для [Telegram канала]. Продукт: [студии растяжки в Калининграде]. Аудитория: [девушки 25-30, которые хотят улучшить форму, самочувствие и жизненный тонус]

Чем больше деталей вы укажите, тем подробнее будет ответ.

<b>Полезный совет:</b> ИИ не задаёт вопросов, пока вы не попросите его — поэтому иногда бывает полезно добавить: "задавай уточняющие вопросы, если необходимо."""
                    ),
                    parse_mode=constants.ParseMode.HTML,
                    chat_id=update.callback_query.from_user.id,
                    reply_markup=InlineKeyboardMarkup(
                        [
                            [
                                InlineKeyboardButton(
                                    f"Далее",
                                    callback_data=f"start_3",
                                ),
                            ]
                        ]
                    ),
                )
            elif stage == "3":
                await update.get_bot().send_message(
                    text=(
                        f"""<b>2. Задавайте правильный контекст</b>

<b>Пример контекста:</b>
Веди себя как учитель английского. Я буду писать тебе фразы на английском, а ты исправляй грамматику и орфографию, и дай совет, как можно улучшить фразу. Моя первая фраза: ... 

После этого бот будет вести себя так, как вы попросили."""
                    ),
                    parse_mode=constants.ParseMode.HTML,
                    chat_id=update.callback_query.from_user.id,
                    reply_markup=InlineKeyboardMarkup(
                        [
                            [
                                InlineKeyboardButton(
                                    f"Далее",
                                    callback_data=f"start_4",
                                ),
                            ]
                        ]
                    ),
                )
            elif stage == "4":
                await update.get_bot().send_message(
                    text=(
                        f"""<b>3. Используйте команды в меню</b>


👉🏻 Команда /reset 

Используйте эту команду, если меняете тему разговора, чтобы очистить историю.

Например, вы получили нужный результат и теперь хотите обсудить другие задачи. Нажмите на эту команду и бот очистит контекст диалога. 

👉🏻 Команда /image 

Генерация картинок через нейросеть Dalle-3. 
Нужно написать команду и описать изображение, которое хотите получить. Можно на русском языке 

Например: /image иллюстрация милого робота с надписью “neuroscribe”

👉🏻 Команда /stats

Это текущая статистика использования вашего ассистента 
Также, внутри этой команды можно выбрать и оплатить тариф 

 👉🏻 Команда /voice 

Это озвучка вашего текста. Пока что у нас есть один голос, но скоро мы добавим еще голоса. Чтобы озвучить текст, напишите команду и вставьте текст для озвучки

Например: /voice Всем привет, сегодня мы обсудим тренды в маркетинге для экспертов. Мой подкаст будет длиться 10 минут, так что приготовьте чай и присаживайтесь поудобнее! 
"""
                    ),
                    parse_mode=constants.ParseMode.HTML,
                    chat_id=update.callback_query.from_user.id,
                    reply_markup=InlineKeyboardMarkup(
                        [
                            [
                                InlineKeyboardButton(
                                    f"Далее",
                                    callback_data=f"start_5",
                                ),
                            ]
                        ]
                    ),
                )
            elif stage == "5":
                await update.get_bot().send_message(
                    text=(
                        f"""<b>4. Учитывайте лимиты использования</b>

На каждом тарифе есть свои ограничения на использование ИИ ассистента. 

Нажмите команду /stats, чтобы посмотреть текущую статистику использования и улучшить тариф. 
"""
                    ),
                    parse_mode=constants.ParseMode.HTML,
                    chat_id=update.callback_query.from_user.id,
                    reply_markup=InlineKeyboardMarkup(
                        [
                            [
                                InlineKeyboardButton(
                                    f"Далее",
                                    callback_data=f"start_6",
                                ),
                            ]
                        ]
                    ),
                )
            elif stage == "6":
                await update.get_bot().send_message(
                    text=(
                        f"""<b>5. Вы можете общаться с ИИ голосом</b>

Просто отправьте ему голосовое сообщение, он транскрибирует и выполнит ваш запрос. 

⚡️Например: “Действуй как опытный копирайтер и маркетолог. Напиши Телеграм-пост для целевой аудитории экспертов разных профессий от лица Максима Наговицына, маркетолога. Напиши 5 ошибок в продвижении в Телеграме в 2024 году” 

Бот расшифрует и напишет текст. Далее вы можете управлять голосом или писать ему запросы в контексте этой задачи. 

Не забудьте, потом нажать /reset для сброса контекста
"""
                    ),
                    parse_mode=constants.ParseMode.HTML,
                    chat_id=update.callback_query.from_user.id,
                    reply_markup=InlineKeyboardMarkup(
                        [
                            [
                                InlineKeyboardButton(
                                    f"Далее",
                                    callback_data=f"start_7",
                                ),
                            ]
                        ]
                    ),
                )
            elif stage == "7":
                await update.get_bot().send_message(
                    text=(
                        f"""Поздравляю, {update.callback_query.from_user.first_name}! 

Теперь вы знаете, как улучшить свою эффективность на 80% вместе с ИИ ассистентом. 

⚡️<b>Не забудьте ознакомиться с книгой-гайдом со 110 задачами для ИИ</b>
https://neuroscribe.ru/110tasks  

⚡️<b>Списком промтов-запросов для бизнес задач</b>
https://telegra.ph/Spisok-promtov-i-zaprosov-dlya-II--nejroskrajb-02-23 

А также подписывайтесь на наш телеграм-канал, где мы выкладываем гайды по использованию ИИ в своих рабочих процессах 
👉 @neuroscribe

У вас осталось 3 дня, чтобы протестировать все функции бота абсолютно бесплатно 

Если что, пишите @maxnagovitsyn 

Высокой продуктивности, {update.callback_query.from_user.first_name}, на связи 👋"""
                    ),
                    parse_mode=constants.ParseMode.HTML,
                    chat_id=update.callback_query.from_user.id,
                )
        if callback_data == "change_rate":
            rates_text = [
                (
                    f"*{idx + 1}. {rate['name']}*\n"
                    f"Токенов GPT-4: {rate['gpt4_rate'] if rate['gpt4_rate'] else 'нет'}\n"
                    f"Токенов GPT-3.5: {rate['gpt35_rate']}\n"
                    f"Изображений: {rate['dalle_rate']}\n"
                    f"Речь в текст: {rate['whisper_rate']} минут\n"
                    f"Озвучка: {rate['tts_rate']} символов\n"
                    f"Стоимость: {rate['price']} рублей / месяц\n"
                    f"----------------------------\n"
                )
                for idx, rate in enumerate(self.rates.values())
            ]
            await update.get_bot().send_message(
                text=(
                        "*Все тарифы:*\n"
                        + "".join(rates_text)
                        + "🙂Пожалуйста, выберите подходящий тариф, и нажмите на одну из кнопок, для перехода к оплате."
                ),
                chat_id=update.callback_query.from_user.id,
                parse_mode=constants.ParseMode.MARKDOWN,
                reply_markup=InlineKeyboardMarkup(
                    [
                        [
                            InlineKeyboardButton(
                                f"{i + 1}",
                                callback_data=f"buy_rate{i}",
                            )
                            for i in range(len(self.rates))
                        ]
                    ]
                ),
            )
        mrh_login = "neuroscribe"
        mrh_pass1 = self.config["robokassa_password"]
        mrh_pass2 = self.config["robokassa_password2"]

        if callback_data.startswith("buy_rate"):
            rate_number = int(callback_data.split("buy_rate")[-1])
            rate = self.rates[list(self.rates.keys())[rate_number]]
            inv_id = random.randint(0, 2 ** 31 - 1)
            inv_desc = f"Покупка тарифа {rate_number}"
            crc = hashlib.md5(
                f"{mrh_login}:{rate['price']}:{inv_id}:{mrh_pass1}".encode()
            ).hexdigest()
            await update.get_bot().send_message(
                chat_id=update.callback_query.from_user.id,
                text=f"Отличный выбор, {update.callback_query.from_user.first_name}!\nВаша ссылка на оплату сформирована. \n\nПожалуйста, нажмите на кнопку ниже, чтобы оплатить",
                reply_markup=InlineKeyboardMarkup(
                    [
                        [
                            InlineKeyboardButton(
                                text="Перейти к оплате",
                                url=f"https://auth.robokassa.ru/Merchant/Index.aspx?MerchantLogin={mrh_login}&OutSum={rate['price']}&InvoiceID={inv_id}&Description={inv_desc}&SignatureValue={crc}",
                            )
                        ]
                    ]
                ),
            )
            await update.get_bot().send_message(
                chat_id=update.callback_query.from_user.id,
                text=f"Обязательно проверьте оплату, нажав кнопку ниже 👇",
                reply_markup=InlineKeyboardMarkup(
                    [
                        [
                            InlineKeyboardButton(
                                text="👍 Оплатил(-а)", callback_data=f"check_pay{inv_id}"
                            )
                        ]
                    ]
                ),
            )

        if callback_data.startswith("check_pay"):
            payment_id = int(callback_data.split("check_pay")[-1])
            signature = hashlib.md5(
                f"{mrh_login}:{payment_id}:{mrh_pass2}".encode()
            ).hexdigest()
            r = requests.get(
                f"https://auth.robokassa.ru/Merchant/WebService/Service.asmx/OpStateExt?MerchantLogin={mrh_login}&InvoiceID={payment_id}&Signature={signature}"
            )
            result = xmltodict.parse(r.content.decode("utf-8"))
            user_id = update.callback_query.from_user.id
            if not result["OperationStateResponse"].get("State"):
                await update.get_bot().send_message(
                    chat_id=update.callback_query.from_user.id,
                    text="Похоже, что-то пошло не так и мы не получили вашу оплату :(",
                    reply_markup=InlineKeyboardMarkup(
                        [
                            [
                                InlineKeyboardButton(
                                    text="Написать в поддержку",
                                    url="https://t.me/maxnagovitsyn",
                                ),
                                InlineKeyboardButton(
                                    text="Проверить оплату",
                                    callback_data=f"check_pay{payment_id}",
                                ),
                            ]
                        ]
                    ),
                )
                return
            if result["OperationStateResponse"]["State"]["Code"] == "100":
                try:
                    if payment_id == int(
                            self.db.get_user(
                                chat_id=update.callback_query.from_user.id
                            ).__dict__["last_pay_id"]
                    ):
                        await update.get_bot().send_message(
                            chat_id=update.callback_query.from_user.id,
                            text="Вы уже оплатили этот счёт.",
                        )
                        return
                except Exception as e:
                    print(e)
                # Всё успешно
                summ = int(float(result["OperationStateResponse"]["Info"]["OutSum"]))
                rate = None
                rate_type = None
                for key in list(self.rates.keys()):
                    r = self.rates[key]
                    if r["price"] == summ:
                        rate = r
                        rate_type = key
                        break
                if rate:
                    self.db.update_user_field(user_id, "gpt4_rate", rate["gpt4_rate"])
                    self.db.update_user_field(user_id, "gpt35_rate", rate["gpt35_rate"])
                    self.db.update_user_field(user_id, "dalle_rate", rate["dalle_rate"])
                    self.db.update_user_field(
                        user_id, "whisper_rate", rate["whisper_rate"]
                    )
                    self.db.update_user_field(user_id, "tts_rate", rate["tts_rate"])
                    self.db.update_user_field(
                        user_id, "rate_end_date", datetime.now() + timedelta(days=30)
                    )
                    self.db.update_user_field(user_id, "rate_type", rate_type)
                    self.db.update_user_field(user_id, "is_free", False)
                    self.db.update_user_field(user_id, "last_pay_id", payment_id)
                    await update.get_bot().send_message(
                        chat_id=update.callback_query.from_user.id,
                        text=f"🥳 Спасибо, {update.callback_query.from_user.first_name}! Оплата успешно прошла, ваш тариф: {rate['name']}\n\nМожете продолжать общение с нейроскрайбом",
                    )
                    if update.callback_query.from_user.username:
                        text = (
                                "Пользователь @"
                                + update.callback_query.from_user.username
                                + f" купил тариф {rate_type} за {summ} руб."
                        )
                    else:
                        chat_id = update.callback_query.from_user.id
                        text = (
                                "Пользователь с chat_id "
                                + str(chat_id)
                                + f" купил тариф {rate_type} за {summ} руб."
                        )

                    await update.get_bot().send_message(
                        chat_id=self.config["admin_group_id"], text=text
                    )
                    return
                else:
                    print(summ)
            else:
                await update.get_bot().send_message(
                    chat_id=update.callback_query.from_user.id,
                    text="Похоже, что-то пошло не так и мы не получили вашу оплату :(",
                )
        try:
            if callback_data.startswith(callback_data_suffix):
                unique_id = callback_data.split(":")[1]
                total_tokens = 0

                # Retrieve the prompt from the cache
                query = self.inline_queries_cache.get(unique_id)
                if query:
                    self.inline_queries_cache.pop(unique_id)
                else:
                    error_message = (
                        f'{localized_text("error", bot_language)}. '
                        f'{localized_text("try_again", bot_language)}'
                    )
                    await edit_message_with_retry(
                        context,
                        chat_id=None,
                        message_id=inline_message_id,
                        text=f"{query}\n\n_{answer_tr}:_\n{error_message}",
                        is_inline=True,
                    )
                    return

                unavailable_message = localized_text(
                    "function_unavailable_in_inline_mode", bot_language
                )
                if self.config["stream"]:
                    stream_response = self.openai.get_chat_response_stream(
                        chat_id=user_id, query=query
                    )
                    i = 0
                    prev = ""
                    backoff = 0
                    async for content, tokens in stream_response:
                        if is_direct_result(content):
                            cleanup_intermediate_files(content)
                            await edit_message_with_retry(
                                context,
                                chat_id=None,
                                message_id=inline_message_id,
                                text=f"{query}\n\n_{answer_tr}:_\n{unavailable_message}",
                                is_inline=True,
                            )
                            return

                        if len(content.strip()) == 0:
                            continue

                        cutoff = get_stream_cutoff_values(update, content)
                        cutoff += backoff

                        if i == 0:
                            try:
                                await edit_message_with_retry(
                                    context,
                                    chat_id=None,
                                    message_id=inline_message_id,
                                    text=f"{query}\n\n{answer_tr}:\n{content}",
                                    is_inline=True,
                                )
                            except:
                                continue

                        elif (
                                abs(len(content) - len(prev)) > cutoff
                                or tokens != "not_finished"
                        ):
                            prev = content
                            try:
                                use_markdown = tokens != "not_finished"
                                divider = "_" if use_markdown else ""
                                text = f"{query}\n\n{divider}{answer_tr}:{divider}\n{content}"

                                # We only want to send the first 4096 characters. No chunking allowed in inline mode.
                                text = text[:4096]

                                await edit_message_with_retry(
                                    context,
                                    chat_id=None,
                                    message_id=inline_message_id,
                                    text=text,
                                    markdown=use_markdown,
                                    is_inline=True,
                                )

                            except RetryAfter as e:
                                backoff += 5
                                await asyncio.sleep(e.retry_after)
                                continue
                            except TimedOut:
                                backoff += 5
                                await asyncio.sleep(0.5)
                                continue
                            except Exception:
                                backoff += 5
                                continue

                            await asyncio.sleep(0.01)

                        i += 1
                        if tokens != "not_finished":
                            total_tokens = int(tokens)

                else:

                    async def _send_inline_query_response():
                        nonlocal total_tokens
                        # Edit the current message to indicate that the answer is being processed
                        await context.bot.edit_message_text(
                            inline_message_id=inline_message_id,
                            text=f"{query}\n\n_{answer_tr}:_\n{loading_tr}",
                            parse_mode=constants.ParseMode.MARKDOWN,
                        )

                        logging.info(f"Generating response for inline query by {name}")
                        response, total_tokens = await self.openai.get_chat_response(
                            chat_id=user_id, query=query
                        )

                        if is_direct_result(response):
                            cleanup_intermediate_files(response)
                            await edit_message_with_retry(
                                context,
                                chat_id=None,
                                message_id=inline_message_id,
                                text=f"{query}\n\n_{answer_tr}:_\n{unavailable_message}",
                                is_inline=True,
                            )
                            return

                        text_content = f"{query}\n\n_{answer_tr}:_\n{response}"

                        # We only want to send the first 4096 characters. No chunking allowed in inline mode.
                        text_content = text_content[:4096]

                        # Edit the original message with the generated content
                        await edit_message_with_retry(
                            context,
                            chat_id=None,
                            message_id=inline_message_id,
                            text=text_content,
                            is_inline=True,
                        )

                    await wrap_with_indicator(
                        update,
                        context,
                        _send_inline_query_response,
                        constants.ChatAction.TYPING,
                        is_inline=True,
                    )

                add_chat_request_to_usage_tracker(
                    self.usage, self.config, user_id, total_tokens
                )

        except Exception as e:
            logging.error(
                f"Failed to respond to an inline query via button callback: {e}"
            )
            logging.exception(e)
            localized_answer = localized_text("chat_fail", self.config["bot_language"])
            await edit_message_with_retry(
                context,
                chat_id=None,
                message_id=inline_message_id,
                text=f"{query}\n\n_{answer_tr}:_\n{localized_answer} {str(e)}",
                is_inline=True,
            )

    async def check_allowed_and_within_budget(
            self, update: Update, context: ContextTypes.DEFAULT_TYPE, is_inline=False
    ) -> bool:
        """
        Checks if the user is allowed to use the bot and if they are within their budget
        :param update: Telegram update object
        :param context: Telegram context object
        :param is_inline: Boolean flag for inline queries
        :return: Boolean indicating if the user is allowed to use the bot
        """
        name = (
            update.inline_query.from_user.name
            if is_inline
            else update.message.from_user.name
        )
        user_id = (
            update.inline_query.from_user.id
            if is_inline
            else update.message.from_user.id
        )

        if not await is_allowed(self.config, update, context, is_inline=is_inline):
            logging.warning(
                f"User {name} (id: {user_id}) is not allowed to use the bot"
            )
            await self.send_disallowed_message(update, context, is_inline)
            return False
        if not is_within_budget(self.config, self.usage, update, is_inline=is_inline):
            logging.warning(f"User {name} (id: {user_id}) reached their usage limit")
            await self.send_budget_reached_message(update, context, is_inline)
            return False

        return True

    async def send_disallowed_message(
            self, update: Update, _: ContextTypes.DEFAULT_TYPE, is_inline=False
    ):
        """
        Sends the disallowed message to the user.
        """
        if not is_inline:
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                text=self.disallowed_message,
                disable_web_page_preview=True,
            )
        else:
            result_id = str(uuid4())
            await self.send_inline_query_result(
                update, result_id, message_content=self.disallowed_message
            )

    async def send_budget_reached_message(
            self, update: Update, _: ContextTypes.DEFAULT_TYPE, is_inline=False
    ):
        """
        Sends the budget reached message to the user.
        """
        if not is_inline:
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update), text=self.budget_limit_message
            )
        else:
            result_id = str(uuid4())
            await self.send_inline_query_result(
                update, result_id, message_content=self.budget_limit_message
            )

    async def post_init(self, application: Application) -> None:
        """
        Post initialization hook for the bot.
        """
        await application.bot.set_my_commands(
            self.group_commands, scope=BotCommandScopeAllGroupChats()
        )
        await application.bot.set_my_commands(self.commands)

    def run(self):
        """
        Runs the bot indefinitely until the user presses Ctrl+C
        """
        application = (
            ApplicationBuilder()
            .token(self.config["token"])
            .proxy_url(self.config["proxy"])
            .get_updates_proxy_url(self.config["proxy"])
            .post_init(self.post_init)
            .concurrent_updates(True)
            .build()
        )

        application.add_handler(
            MessageHandler(filters.PHOTO | filters.Document.IMAGE, self.vision)
        )

        application.add_handler(CommandHandler("reset", self.reset))
        application.add_handler(CommandHandler("help", self.help))
        application.add_handler(CommandHandler("dump", self.dump))
        application.add_handler(CommandHandler("mail", self.mail))
        application.add_handler(CommandHandler("image", self.image))
        application.add_handler(CommandHandler("voice", self.tts))
        application.add_handler(CommandHandler("model", self.model))
        application.add_handler(CommandHandler("assistant", self.assistant))
        application.add_handler(CommandHandler("support", self.support))
        application.add_handler(CommandHandler("sdxl", self.sdxl))
        application.add_handler(CommandHandler("bg", self.bg))
        application.add_handler(CommandHandler("sticker", self.sticker))
        application.add_handler(CommandHandler("pay", self.pay))
        application.add_handler(CommandHandler("start", self.start))
        application.add_handler(CommandHandler("stats", self.stats))
        application.add_handler(CommandHandler("resend", self.resend))
        application.add_handler(CommandHandler("change_rate", self.change_rate))
        application.add_handler(CommandHandler("admin", self.admin))
        application.add_handler(CommandHandler("keys", self.keys))
        application.add_handler(CommandHandler("keys_get", self.keys_get))
        # application.add_handler(CommandHandler("keys_balance", self.keys_balance))
        application.add_handler(CommandHandler("keys_delete", self.keys_delete))
        application.add_handler(CommandHandler("keys_add", self.keys_add))
        application.add_handler(
            CommandHandler(
                "chat",
                self.prompt,
                filters=filters.ChatType.GROUP | filters.ChatType.SUPERGROUP,
            )
        )
        application.add_handler(
            MessageHandler(
                filters.AUDIO
                | filters.VOICE
                | filters.Document.AUDIO
                | filters.VIDEO
                | filters.VIDEO_NOTE
                | filters.Document.VIDEO,
                self.transcribe,
            )
        )
        application.add_handler(
            MessageHandler(filters.TEXT & (~filters.COMMAND), self.prompt)
        )
        application.add_handler(
            InlineQueryHandler(
                self.inline_query,
                chat_types=[
                    constants.ChatType.GROUP,
                    constants.ChatType.SUPERGROUP,
                    constants.ChatType.PRIVATE,
                ],
            )
        )
        application.add_handler(CallbackQueryHandler(self.handle_callback_inline_query))

        # application.add_error_handler(error_handler)

        application.run_polling()
