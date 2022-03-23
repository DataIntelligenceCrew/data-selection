from itertools import islice


model_details = {}

with open('./test.txt', 'r') as ifile:
  n = 4
  for line in ifile:
    if line.startswith('Model ID:'):
      model_details["details"] = list(islice(ifile, n))[2:].append(line.strip()) 
ifile.close()


for key, value in model_details.items():
  print(key + ' : ' + str(value) + '\n')
