import telepot
from bs4 import BeautifulSoup

def telegram_notify(msg):
    token = "178350795:AAFG7yae2SSt52GLek2bKS43oK7BaywWxRw"
    bot = telepot.Bot(token)
    b = bot.getMe()
    response = bot.getUpdates()
    bot.sendMessage(31747780, msg)

import scrapy
from scrapy.crawler import CrawlerProcess
import json
import os
import re

from scrapy.settings import Settings

DATADIR = os.environ["HOME"] + "/data/allen-ai-challenge/"


def get_h_all_text(doc):
    start = []
    names = []
    regex = re.compile(r"<h[1-6].+\n.*")
    for match in regex.finditer(doc):
        h_name = re.findall(r">\n.*", match.group())[0][2:].strip()
        names.append(h_name)
        start.append(match.start())

    s = start[0] #put 2 to skip first two h
    paragraphs = {}
    for i in range(1, len(start)): #put 3 to skip first two h
        e = start[i] - 1
        if names[i - 1] not in ['Practice', 'Questions', 'Review', 'Explore More',
                              'Explore More I','Explore More II','Explore More III' 'References']:
            paragraphs[str(i) + "_" + names[i - 1]] = (BeautifulSoup(doc[s:e], 'html.parser').get_text())
        s = e + 1
    return paragraphs


def get_h4_text(doc):
    doc = BeautifulSoup(doc, 'html.parser')
    doc_name = doc.find_all('title')[0].text.strip().replace("?", "")

    topics = {}
    for topic in doc.find_all('h4'):
        topic_name = topic.text.strip().replace("?", "")
        if topic_name.lower() in ['practice', 'questions', 'review', 'explore more', 'references']:
            continue
        content = [""]
        for p in topic.find_next_siblings():
            if p.text.strip().lower() in ['practice', 'questions', 'review', 'explore more', 'references']:
                break
            #add
            content += p.text.split()
        topics[topic_name] = ' '.join(content)

    return topics


class Article(scrapy.Item):
    science = scrapy.Field()
    concept = scrapy.Field()
    contents = scrapy.Field()
    url = scrapy.Field()


class Concept(scrapy.Item):
    concept = scrapy.Field()


class JsonWriterPipeline(object):

    def process_item(self, item, spider):
        science = item["science"]
        concept = item["concept"]
        filename = "%s/%s_%s.json" % (DATADIR, science, concept)
        with open(filename, "w") as f:
            json.dump(dict(item), f)
        return item


class ConceptJsonWriterPipeline(object):

    def __init__(self):
        self.file = open('concepts.json', 'wb')

    def process_item(self, item, spider):
        concept = item["concept"]
        line = json.dumps(concept) + "\n"
        self.file.write(line)
        return item


class MySpider(scrapy.Spider):
    name = 'ck12'
    allowed_domains = ['ck12.org']
    start_urls = [
        'https://www.ck12.org/earth-science/',
        'https://www.ck12.org/life-science/',
        'https://www.ck12.org/physical-science/',
        'https://www.ck12.org/biology/',
        'https://www.ck12.org/chemistry/',
        'https://www.ck12.org/physics/'
    ]
    custom_settings = {'DOWNLOAD_HANDLERS': {'s3': None}}

    def parse(self, response):
        concepts = []
        for concept_path in response.xpath('//li[@class="concepts"]/a/@href').extract():
            science, concept = concept_path[1:-1].split("/")
            concepts.append(concept)
            concept_urls = ["%s/%s/lesson/%s" % (response.url, concept, concept),
                            "%s/%s/lesson/%s" % (response.url, concept, concept.split("-in-")[0]),
                            "%s/%s/lesson/%s" % (response.url, concept, concept.split("-in-")[0] + "-Basic")]
        concept = Concept()
        concept["concept"] = [concepts]
        return concept
            # for url in concept_urls:
            #     request = scrapy.Request(url, self.parse_page)
            #     request.meta["science"] = science
            #     request.meta["concept"] = concept
            #     yield request


    def parse_page(self, response):
        print(response.url)
        details_path = response.xpath('//div[@id="modality_content"]/@data-loadurl')[0]._root
        details_url = "http://www.ck12.org%s" % details_path
        request = scrapy.Request(details_url, self.parse_article)
        request.meta["science"] = response.meta["science"]
        request.meta["concept"] = response.meta["concept"]
        yield request

    def parse_article(self, response):
        article = Article()
        article["science"] = response.meta["science"]
        article["concept"] = response.meta["concept"]
        article["contents"] = get_h_all_text(response.body)
        article["url"] = response.url
        return article


settings = Settings()
settings.set('ITEM_PIPELINES', {
    '__main__.ConceptJsonWriterPipeline': 100
})

# configure_logging({'LOG_FORMAT': '%(levelname)s: %(message)s'})
crawler = CrawlerProcess(settings)

crawler.crawl(MySpider)
crawler.start()

telegram_notify("done")