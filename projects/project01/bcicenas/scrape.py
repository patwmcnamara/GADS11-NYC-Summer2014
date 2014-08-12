import os, sys, random
from time import sleep,time
import json
import urllib2
from parse import parse
from bs4 import BeautifulSoup

ifile = 'sneaker_freaker_all.html'
ofile = open('sneakers.json', 'w')

def fetch_url(url):
    stime = time()
    try: 
        data = urllib2.urlopen(urllib2.Request(model_url)).read()
        print('fetched %s in %s seconds') % (model_url, (time() - stime))
    except:
        pass
        return None 
    return data

def get_model_urls(f):
    model_urls = []
    data = open(f).read()
    soup = BeautifulSoup(data)
    for link in soup.find_all('a'):
        if link.get('href')[41:46] == 'model':
            model_urls.append(link.get('href'))
    random.shuffle(model_urls)
    return model_urls

for model_url in get_model_urls(ifile):
    model = {}
    manf = parse('http://archive.sneakerfreaker.com/museum/model/{}', (model_url.split("-")[0]))
    model['manufacturer'] = manf[0]
    data = fetch_url(model_url)
if data:
soup = BeautifulSoup(data)
for li in soup.findChildren('li', "clearfix"):
attrib = li.getText().split(":")
model[attrib[0].strip('\n')] = attrib[1].strip('\t')
ofile.write(json.dumps(model))
else:
print('failed to fetch %s') % (model_url)
    sleep((random.randint(1, 10)))
ofile.close()
