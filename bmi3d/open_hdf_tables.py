import tables
import numpy
import matplotlib.pyplot as plt

#replace this with your hdf filename
#fname = 'c:\\Users\\Si Jia\\AppData\\Local\\Temp\\tmp9fswwtwp.h5'
fname = '/tmp/tmpdcbqn2zo.h5'
hdffile = tables.open_file(fname,'r') #read-onl

print(hdffile)

#get table information
# more methods refer to this 
# https://www.pytables.org/usersguide/libref/structured_storage.html#tables.Table
table = hdffile.root.task
print(table.description)

#look at cursor trajectory
cursor_coor = table.col('cursor')
plt.plot(cursor_coor[:,0], cursor_coor[:,2])
plt.show()



