import pandas as pd

frame = pd.read_csv ('test.txt', header=None, delim_whitespace=True)
values = frame.values
j = -1
for i in range(values.shape[0]):
	if i % 3 == 0:
		j += 1
		
		
	print('ATOM   %4d   %s %s%4d      %8.3f %7.3f %7.3f  1.00  0.00           %s    0.00000' %(values[i,1],values[i,2], values[i,3],j,values[i,5],values[i,6],values[i,7], values[i,10]))


# print(string)
