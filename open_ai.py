from openai import OpenAI


class OpenAi:
  def __init__(self):
    self.client = OpenAI()

  def upload_file(self, filename):
    self.client.files.create(
        file=open("data/openai/dua lipa - Google Search.html", "rb"),
        purpose="assistants"
    )

  def process(self, inp):
    response = self.client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role": "system", "content": "You create queries to search on google. user writes input you create a query to search on google which will give you the answer."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "2020 world series winner."},
        {"role": "user", "content": "reading knowledge card entires in google search results."},
      ]
    )
    print(response)
    pass
