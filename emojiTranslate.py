import emoji


#print(emoji.demojize('Python is 😱😊😄😉😆😋😁'))

'''
with open('/home/mimi/Desktop/PFE/DATASETS/algerianYoutubeDataset.txt', 'r') as f:
    dataset_list = [line.strip() for line in f]
print(len(dataset_list))'''

dataLIst=[
    'Ana lowllaaa li chefto wllh ou9ssim billah 🙈🙈😍😍',
    '1 er vue l3leeemm 😂😍'
]

for comment in dataLIst :
     print(emoji.demojize(comment))