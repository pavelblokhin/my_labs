import scrapy
import sys
from datetime import datetime
from steam_parsing.items import SteamParsingItem


class IndiSpider(scrapy.Spider):
    name = 'steam'
    allowed_domains = ['store.steampowered.com']
    # start_urls = [
    #     'https://store.steampowered.com/search/?term=%D0%B8%D0%BD%D0%B4%D0%B8',
    #     'https://store.steampowered.com/search/?term=%D1%88%D1%83%D1%82%D0%B5%D1%80&supportedlang=russian&ndl=1',
    #     'https://store.steampowered.com/search/?term=%D1%81%D1%82%D1%80%D0%B0%D1%82%D0%B5%D0%B3%D0%B8%D0%B8&ignore_preferences=1&ndl=1'
    #     ]
    # признаюсь я попросил ссылки у одногрупнка, потому что по моим ссылкам много игр с ограничениями и они н парсятся
    start_urls = [
        'https://store.steampowered.com/search/?term=шутер&ndl=1&ignore_preferences=1',
        'https://store.steampowered.com/search/?term=карты&ndl=1&ignore_preferences=1',
        'https://store.steampowered.com/search/?term=инди&ndl=1&ignore_preferences=1'
    ]
    
    def parse_first(self, response):
        for i in range(1, 3):
            url = response + f"&page={i}"
            yield scrapy.Request(url=url, callback=self.parse)
        
    def parse(self, response):
        game_links = response.xpath('//a[contains(@class, "search_result_row")]/@href').getall()
        for link in game_links:
            yield scrapy.Request(link, callback=self.parse_game)
        
    
    def parse_game(self, response):
        
        
        name = response.css('div.apphub_AppName::text').get()
        category = response.css('div.blockbg a::text').getall()[1:]
        overall_assessment = response.xpath("//div[contains(@class, 'user_reviews_summary_row')][2]//span[contains(@class, 'game_review_summary')]/text()").get()
        
        reviews = response.xpath("//div[contains(@class, 'user_reviews_summary_row')][2]//span[@class='responsive_hidden']/text()").get()
        if reviews == None:
            reviews = response.xpath("//div[contains(@class, 'user_reviews_summary_row')]//span[@class='responsive_hidden']/text()").get()
            overall_assessment = response.xpath("//div[contains(@class, 'user_reviews_summary_row')]//span[contains(@class, 'game_review_summary')]/text()").get()
        if reviews:
            reviews = reviews.split()
        else:
            reviews = None
            
        date = response.css('div.date::text').get()
        release = datetime.strptime(date, "%d %b, %Y")
        developer = response.xpath("//div[@id='developers_list']/a/text()").get()
        tags = response.xpath('//a[@class="app_tag"]/text()').getall()
            
        price = response.xpath("//div[contains(@class, 'game_purchase_price') and contains(@class, 'price')]/text()").get()
        if price == None:
            price = response.xpath("//div[@class='discount_final_price']/text()").get()
        if price:
            price = price.strip()
        else:
            price = None
            
        platforms = []
        if response.css('.platform_img.win'):
            platforms.append('Windows')
        if response.css('.platform_img.mac'):
            platforms.append('Mac')
        if response.css('.platform_img.linux'):
            platforms.append('Linux')

        
        item = SteamParsingItem()
        item['name'] = name
        item['category'] = category
        item['reviews'] = reviews
        item['overall_assessment'] = overall_assessment
        item['release'] = release
        item['developer'] = developer
        item['tags'] = ' '.join(map(lambda x: x.strip(), tags))
        item['price'] = price
        item['platforms'] = platforms
        yield item