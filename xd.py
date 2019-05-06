import requests,json
videoId = '-K89GvM4FsA'
url = 'https://www.googleapis.com/youtube/v3/commentThreads?key=AIzaSyCRo_i-DpYUKlKkeXTLm6L-ldQksi_urAw&textFormat=plainText&part=snippet&videoId='+videoId+'&maxResults=100'
# pageToken=
resp = requests.get(url=url)
data = resp.json()
f= open("/home/mimi/Desktop/PFE/DATASETS/textdata_"+videoId+".txt","w+")
for item in data['items']:
        text = item['snippet']['topLevelComment']['snippet']['textDisplay'].replace('\n','')
        f.write(text +'\r\n')
while 'nextPageToken' in data.keys():
    nextPageToken = data['nextPageToken']
    url2 = url + '&pageToken='+nextPageToken
    resp = requests.get(url=url2)
    data = resp.json()
    for item in data['items']:
        text = item['snippet']['topLevelComment']['snippet']['textDisplay'].replace('\n','')
        f.write(str(text) +'\r\n')
f.close()