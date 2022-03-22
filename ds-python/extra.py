from itertools import islice


model_details = {}

with open('./test.txt', 'r') as ifile:
  n = 4
  for line in ifile:
    if line.startswith('Model ID:'):
      model_details[line] = list(islice(ifile, n))
ifile.close()


print(model_details)