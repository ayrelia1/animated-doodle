from typing import Dict
import os
from duckduckgo_search import DDGS

from .plugin import Plugin

import json
import random

class DDGTranslatePlugin(Plugin):
    """
    A plugin to translate a given text from a language to another, using DuckDuckGo
    """

    def get_source_name(self) -> str:
        return "DuckDuckGo Translate"

    def get_spec(self) -> [Dict]:
        return [
            {
                "name": "translate",
                "description": "Translate a given text from a language to another",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The text to translate",
                        },
                        "to_language": {
                            "type": "string",
                            "description": "The language to translate to (e.g. 'it')",
                        },
                    },
                    "required": ["text", "to_language"],
                },
            }
        ]

    async def execute(self, function_name, helper, **kwargs) -> Dict:
        
        proxies = os.getenv("PROXIES")
        data = json.loads(proxies)
        proxies = data['proxies']
        random_proxy = random.choice(proxies)
        
        
        with DDGS(proxies=f'socks5://{random_proxy['proxy']}') as ddgs: # коннектим прокси
            return ddgs.translate(kwargs["text"], to=kwargs["to_language"])
