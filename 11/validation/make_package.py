import sys
import glob
import os

import argparse

parser = argparse.ArgumentParser(description='Run FAT online packaging.')
parser.add_argument('path', type=str, default='../cxxexample',
                    help='fat implementation path')
parser.add_argument('--shortname', type=str, default='',
                    help='shortname')
parser.add_argument('--number', type=int, default='',
                    help='submission number, like 0,1,2...')
parser.add_argument('--email', type=str, default='',
                    help='participate email')
args = parser.parse_args()

package_files = list(glob.glob('%s/pyfat_implement*'%args.path))
package_files.append('%s/assets'%args.path)

filename = '%s_v%02d' % (args.shortname, args.number)

cmd = "tar -cvf %s.tar %s"%(filename, " ".join(package_files))
print(cmd)
os.system(cmd)
#cmd = 'wget -O ./fat_11_public.gpg https://raw.githubusercontent.com/nelivacn/FAT/master/files/fat_11_public.gpg'
#os.system(cmd)
cmd = 'gpg --import ../../files/fat_11_public.gpg'
os.system(cmd)
cmd = 'gpg --default-key %s --output %s.tar.gpg --encrypt --recipient fat@zhongdun.com.cn --sign %s.tar' % (args.email, filename, filename)
os.system(cmd)
cmd = 'gpg --armor --output %s.gpg.key --export %s' % (args.shortname, args.email)
os.system(cmd)
cmd = 'tar -czf %s.tar.gz %s.tar.gpg %s.gpg.key' % (filename, filename, args.shortname)
os.system(cmd)

print('Please submit %s.tar.gz'%filename)

