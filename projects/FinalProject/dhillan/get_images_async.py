
import asyncio
import aiohttp
import math
import time
import urllib.request

def generate_url_list(base_url, year_start):
  url_list = []
  file_list = []
  for year in range(year_start,2014):
    year_str = str(year)
    month_start = 1
    if year == 2010:
      month_start = 5
    for month in range(month_start,13):
      month_str = str(month)
      if month<10:
        month_str = '0'+ month_str

      for day in range(1,32):
        day_str = str(day)
        if day<10:
          day_str = '0'+ day_str

        for hour in range(0,24):
          hour_str= str(hour)
          if hour<10:
            hour_str = '0'+ hour_str        

          for minute_str in ['0000', '1500', '3000', '4500']:
            file_str = year_str + month_str + day_str + '_' + hour_str + minute_str + '_M_256.jpg'
            file_list.append(file_str)
            url_list.append(base_url + year_str + '/' + month_str + '/' + day_str + '/' +  file_str)
  return url_list, file_list

@asyncio.coroutine
def save_to_file(img, file_name):
  # file_idx = idx + 10;
  # file_name = file_list[file_idx]
  print(type(file_name))
  print("printing file name: " + file_name)
  f = open('./data/all_256/' + file_name,'wb')
  f.write(img)
  f.close()

@asyncio.coroutine
def fetch(url):
  data = ""
  try:
    yield from asyncio.sleep(1)
    response = yield from aiohttp.request('GET', url)
  except Exception as exc:
      print("ERROR ", url, 'has error', repr(str(exc)))
  else:
      data = (yield from response.read())
      print()
      response.close()

  return data

@asyncio.coroutine
def fetch_image(url, file_name):
    img = yield from fetch(url)

    # save image to file
    yield from save_to_file(img, file_name)

@asyncio.coroutine
def process_batch_of_urls(urls, file_names):
  coros = []
  for url, file_name in zip(urls, file_names):
    coros.append(asyncio.Task(fetch_image(url, file_name)))

  yield from asyncio.gather(*coros)

@asyncio.coroutine
def process_all():
  print('### Started ###')
  start_time = time.time()
  url_list, file_list = generate_url_list('http://jsoc.stanford.edu/data/hmi/images/', 2010)

  pool_size = 200
  url_chunks=[url_list[x:x+pool_size] for x in range(0, len(url_list), pool_size)]
  file_chunks=[file_list[x:x+pool_size] for x in range(0, len(file_list), pool_size)]

  for urls, file_names in zip(url_chunks, file_chunks):
    yield from process_batch_of_urls(urls, file_names)

  total_time = time.time() - start_time
  print("Total number of api requests %d" % len(url_list))
  print("Total time taken to process api requests is %s seconds" % total_time)

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    # connector stores cookies between requests and uses connection pool
    # connector = aiohttp.TCPConnector(share_cookies=True, loop=loop)
    loop.run_until_complete(process_all())