# @Author: gunjianpan
# @Date:   2018-11-11 19:11:19
# @Last Modified by:   gunjianpan
# @Last Modified time: 2018-11-20 20:41:17

cp data.txt test
## Set Tag Where File End
echo '1' >> test

# According article to jonit string
# @1: Remove blank lines
# @2: split by ‘/x’ or ‘/xx’
#     then, jonit string until into another article(recognition by $1)
# @return: Plain text which one plain one artile
sed -n '/./p' test|awk '{split($0,a,"/. |/.. ");b="";for(i=3;i<length(a);i++){b=b" "a[i];}if(!last||last==substr($1,0,15)){total=total""b;}else{print substr(total,2);total=b;}last=substr($1,0,15)}' >> test2

# Remove Chinese punctuation
# @1: replace punctuation Chinese or English to blank
# @2: replace mutil-blank to one-blank
sed 's/[；：，。（）？！《》【】{}“”、——;:,.()?!_]/ /g' test2|sed 's/[ ][ ]*/ /g' >>test3
