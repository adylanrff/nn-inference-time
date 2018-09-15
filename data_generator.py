import sys
import random
import pandas as pd
import numpy as np

RANDOM_RANGE = 2**8 

def generate_random_array(columns,rows):
  data = []
  for _ in range(rows):
    row = [random.randrange(1,RANDOM_RANGE) for _ in range(columns)]
    data.append(row)

  return data

def array_to_csv(data, filepath):
  np_data = np.array(data)
  df = pd.DataFrame(np_data)
  df.to_csv(filepath, header=None, index=None)

if (__name__ == '__main__'):
  arguments = sys.argv[1:]
  
  if (len(arguments) != 3):
    print("Wrong arguments, should be 3")
    print("<columns> <rows> <filepath>")
  else:
    columns = int(arguments[0])
    rows = int(arguments[1])
    filepath = str(arguments[2])

    data = generate_random_array(columns,rows)
    array_to_csv(data,filepath)







