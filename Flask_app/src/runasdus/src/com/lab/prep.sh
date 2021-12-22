set -e                   # Abort on an error
res=$(find * -maxdepth 0 | wc -l)
echo "found $res results"
i=1

for file in *.html; do      
  printf "\r $i"
  ((i++))
  dir=$(basename -s .html ${file})      
  mkdir -p -- "$dir"     
  mv -- "$file" "$dir/priv.html" 
done