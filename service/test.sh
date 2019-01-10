# @Author: gunjianpan
# @Date:   2019-01-01 10:21:20
# @Last Modified by:   gunjianpan
# @Last Modified time: 2019-01-01 10:49:04

if [ ! -d "imgold/$(date -d yesterday +%y/%m/%d)" ];then
    mkdir -p "imgold/$(date -d yesterday +%y/%m/%d)"
else
    echo "Folder have existed."
fi

mv img/*`date -d yesterday +%y-%m-%d`.* "imgold/$(date -d yesterday +%y/%m/%d)"
