# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class SteamParsingItem(scrapy.Item):
    name = scrapy.Field()
    category = scrapy.Field()
    reviews = scrapy.Field()
    overall_assessment = scrapy.Field()
    release = scrapy.Field()
    developer = scrapy.Field()
    tags = scrapy.Field()
    price = scrapy.Field()
    platforms = scrapy.Field()
