import os
import re
import random
import urllib.request
import urllib.parse
import urllib.error
from concurrent.futures import ThreadPoolExecutor

from . import state
from .utils import utfprint


def websearch(query):
    # sanitize query
    query = re.sub(r'[+\-\"\\/*^|<>~`]', '', query) # Remove blacklisted characters
    query = re.sub(r'\s+', ' ', query).strip() # Replace multiple spaces with a single space
    if not query or query=="":
        return []
    query = query[:300] # only search first 300 chars, due to search engine limits
    if query==state.websearch_lastquery:
        print("Returning cached websearch...")
        return state.websearch_lastresponse
    import difflib
    from html.parser import HTMLParser
    num_results = 3
    searchresults = []
    utfprint("Performing new websearch...",1)

    def fetch_searched_webpage(url, random_agent=False):
        from urllib.parse import quote, urlsplit, urlunsplit
        uagent = 'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)'
        if random_agent:
            agents = ["Mozilla/5.0 (Macintosh; Intel Mac OS X 13_2) Gecko/20100101 Firefox/114.0",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.1823.79 Safari/537.36 Edg/114.0.1823.79",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.5938.132 Safari/537.36"]
            uagent = random.choice(agents)
        if state.args.debugmode:
            utfprint(f"WebSearch URL: {url}")
        # Encode non-ASCII parts of the URL
        try:
            split_url = urlsplit(url)
            encoded_path = quote(split_url.path)
            encoded_url = urlunsplit((split_url.scheme, split_url.netloc, encoded_path, split_url.query, split_url.fragment))

            ssl_cert_dir = os.environ.get('SSL_CERT_DIR')
            if not ssl_cert_dir and not state.nocertify and os.name != 'nt':
                os.environ['SSL_CERT_DIR'] = '/etc/ssl/certs'

            req = urllib.request.Request(encoded_url, headers={'User-Agent': uagent})
            with urllib.request.urlopen(req, timeout=30) as response:
                html_content = response.read().decode('utf-8', errors='ignore')
                return html_content
        except urllib.error.HTTPError: #we got blocked? try 1 more time with a different user agent
            try:
                req = urllib.request.Request(encoded_url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36'})
                with urllib.request.urlopen(req, timeout=30) as response:
                    html_content = response.read().decode('utf-8', errors='ignore')
                    return html_content
            except Exception as e:
                utfprint(f"Error fetching text from URL {url}: {e}",1)
                return ""
        except Exception as e:
            utfprint(f"Error fetching text from URL {url}: {e}",1)
            return ""
    def fetch_webpages_parallel(urls):
        with ThreadPoolExecutor() as executor:
            # Submit tasks and gather results
            results = list(executor.map(fetch_searched_webpage, urls))
        return results

    def normalize_page_text(text):
        text = re.sub(r'\s+([.,!?])', r'\1', text)  # Remove spaces before punctuation
        # text = re.sub(r'([.,!?])([^\s])', r'\1 \2', text) # Ensure a single space follows punctuation, if not at the end of a line
        return text

    class VisibleTextParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.texts = []
            self.is_script_or_style = False
        def handle_starttag(self, tag, attrs):
            if tag in {'script', 'style'}:
                self.is_script_or_style = True
        def handle_endtag(self, tag):
            if tag in {'script', 'style'}:
                self.is_script_or_style = False
        def handle_data(self, data):
            if not self.is_script_or_style and data.strip():
                self.texts.append(data.strip())
        def get_text(self):
            return ' '.join(self.texts)

    class ExtractResultsParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.titles = []
            self.urls = []
            self.descs = []
            self.recordingTitle = False
            self.recordingUrl = False
            self.recordingDesc = False
            self.currsegmenttxt = ""

        def handle_starttag(self, tag, attrs):
            if tag == "a":
                # Check if the "class" attribute matches the target class
                for attr_name, attr_value in attrs:
                    if not self.recordingTitle and attr_name == "class" and "result__a" in attr_value.split():
                        self.recordingTitle = True
                        self.currsegmenttxt = ""
                    if not self.recordingUrl and attr_name == "class" and "result__url" in attr_value.split():
                        self.recordingUrl = True
                        self.currsegmenttxt = ""
                    if not self.recordingDesc and attr_name == "class" and "result__snippet" in attr_value.split():
                        self.recordingDesc = True
                        self.currsegmenttxt = ""

        def handle_endtag(self, tag):
            if tag == "a" and self.recordingTitle:
                self.recordingTitle = False
                self.titles.append(self.currsegmenttxt.strip())
                self.currsegmenttxt = ""
            if tag == "a" and self.recordingUrl:
                self.recordingUrl = False
                self.urls.append(f"https://{self.currsegmenttxt.strip()}")
                self.currsegmenttxt = ""
            if tag == "a" and self.recordingDesc:
                self.recordingDesc = False
                self.descs.append(self.currsegmenttxt.strip())
                self.currsegmenttxt = ""

        def handle_data(self, data):
            if self.recordingTitle or self.recordingDesc or self.recordingUrl:
                self.currsegmenttxt += data

    encoded_query = urllib.parse.quote(query)
    search_url = f"https://html.duckduckgo.com/html/?q={encoded_query}"

    try:
        search_html = fetch_searched_webpage(search_url, random_agent=True)
        parser = ExtractResultsParser()
        parser.feed(search_html)
        titles = parser.titles[:num_results]
        searchurls = parser.urls[:num_results]
        descs = parser.descs[:num_results]

        if len(descs)==0 or len(titles)==0 or len(descs)==0:
            utfprint("No results found! Maybe something went wrong...",1)
            return []

        fetchedcontent = fetch_webpages_parallel(searchurls)
        for i in range(len(descs)):
            # dive into the results to try and get even more details
            title = titles[i]
            url = searchurls[i]
            desc = descs[i]
            pagedesc = ""
            try:
                desclen = len(desc)
                html_content = fetchedcontent[i]
                parser2 = VisibleTextParser()
                parser2.feed(html_content)
                scraped = parser2.get_text().strip()
                scraped = normalize_page_text(scraped)
                desc = normalize_page_text(desc)
                s = difflib.SequenceMatcher(None, scraped.lower(), desc.lower(), autojunk=False)
                matches = s.find_longest_match(0, len(scraped), 0, desclen)
                if matches.size > 100 and desclen-matches.size < 100: #good enough match
                    # expand description by some chars both sides
                    expandamtbefore = 200
                    expandamtafter = 800
                    startpt = matches.a - expandamtbefore
                    startpt = 0 if startpt < 0 else startpt
                    endpt =  matches.a + expandamtafter + desclen
                    pagedesc = scraped[startpt:endpt].strip()
            except Exception:
                pass
            searchresults.append({"title":title,"url":url,"desc":desc,"content":pagedesc})

    except Exception as e:
        utfprint(f"Error fetching URL {search_url}: {e}",1)
        return []
    if len(searchresults) > 0:
        state.websearch_lastquery = query
        state.websearch_lastresponse = searchresults
    return searchresults
