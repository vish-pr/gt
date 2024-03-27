import heapq
from datetime import datetime
from pydoc import doc
import stat
from typing import Callable, List, Mapping
from urllib.parse import ParseResult, urljoin, urlparse, urlunparse

from pydantic_core import Url
import requests
from attr import dataclass
from bs4 import BeautifulSoup
from dateutil import parser

# from evals.evals.evals import data  # TODO remove this dependency by making an LLM call


@dataclass
class Document:
  url: str
  parsed_url: ParseResult
  rating: float
  title: str | None
  date: datetime | None
  text: str | None
  outgoing_links: List[str]
  scraped: bool = False

  @staticmethod
  def from_url(url: str):
    parsed_url = urlparse(url)
    return Document(url, parsed_url, UrlManager.rate(parsed_url), None, None, None, [], False)


class UrlManager:
  allowed_domains: List[str] = []

  @classmethod
  def rate(cls, url: ParseResult) -> float:
    if url.netloc not in cls.allowed_domains:
      return -1
    return 0

  @staticmethod
  def get_url_hash(url: ParseResult) -> str:
    return url.netloc + url.path


def news_scraper(doc: Document) -> None:
  if doc.scraped:
    return
  url = doc.url
  r = requests.get(url)
  soup = BeautifulSoup(r.text, 'html.parser')
  doc.scraped = True
  doc.outgoing_links = [urljoin(url, link['href']) for link in soup.find_all('a', href=True)]

  title_element = soup.find('title')
  doc.title = title_element.text if title_element else None

  time_element = soup.find('span', text=lambda x: x is not None and x.text.startswith('Updated: '))
  doc.date = parser.parse(time_element.text.split('Updated: ')[1]) if time_element else None
  text_element = soup.find('div', attrs={'class': '_s30J clearfix'})
  doc.text = text_element.text if text_element else None


class PriorityQueue:
  def __init__(self):
    self._queue = []
    self._index = 0  # To handle cases where the priorities are the same

  def push(self, url: str):
    doc = Document.from_url(url)
    heapq.heappush(self._queue, (-doc.rating, self._index, doc))
    self._index += 1

  def pop(self) -> Document | None:
    try:
      top = heapq.heappop(self._queue)[-1]
    except IndexError:
      return None
    return top


class WebCrawler:
  def __init__(self, start_url: str, scraper: Callable[[Document], None]):
    parsed_url = urlparse(start_url)
    UrlManager.allowed_domains.append(parsed_url.netloc)
    self.added_url_hash = set()
    self.queue = PriorityQueue()
    self.queue.push(start_url)
    self.added_url_hash.add(UrlManager.get_url_hash(parsed_url))
    self.scraper = scraper

  def crawl(self):
    while (doc := self.queue.pop()):
      if doc.rating < 0:
        continue
      self.scraper(doc)
      print(doc)
      for link in doc.outgoing_links:
        link_hash = UrlManager.get_url_hash(urlparse(link))
        if link_hash in self.added_url_hash:
          continue
        self.queue.push(link)
