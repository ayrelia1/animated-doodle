import time
import requests


class Replicate:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def post(self, data: dict):
        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(
            "https://api.replicate.com/v1/predictions", headers=headers, json=data
        )
        return response.json()

    def get(self, url: str):
        response = requests.get(
            url,
            headers={
                "Authorization": f"Token {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        return response.json()

    def run(
        self,
        version: str,
        input: dict,
    ):
        response = self.post({"version": version, "input": input})
        get_url = response["urls"]["get"]
        working = True

        while working:
            response = self.get(get_url)
            print(response)

            if response["status"] == "succeeded":
                return response["output"]
            elif response["status"] == "processing" or response["status"] == "starting":
                time.sleep(5)
            else:
                raise Exception(response)

        return response
