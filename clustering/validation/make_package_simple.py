import sys
import glob
import os

import argparse

parser = argparse.ArgumentParser(description='Run FAT online packaging.')
parser.add_argument('path', type=str, default='../pyexample',
                    help='fat implementation path')
parser.add_argument('--shortname', type=str, default='insightface',
                    help='shortname')
parser.add_argument('--number', type=int, default=0,
                    help='submission number, like 0,1,2...')
args = parser.parse_args()

package_files = list(glob.glob('%s/pyfat_implement*'%args.path))
package_files.append('%s/assets'%args.path)

final_name = 'Clustering_%s_v%02d' % (args.shortname, args.number)

cmd = "tar -czvf %s.tar.gz %s" % (final_name, " ".join(package_files))
print(cmd)
os.system(cmd)

print('Please submit %s.tar.gz' % final_name)
