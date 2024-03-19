import sys
import os

# Add the original folder to sys.path
curr_dir = sys.path[0]
print('curr_dir:', curr_dir)  # temp
src_dir = os.path.join(sys.path[0], 'src')
print('src_dir:', src_dir)  # temp  

sys.path.append(curr_dir)
sys.path.append(src_dir)

for path in sys.path:
    print(path)