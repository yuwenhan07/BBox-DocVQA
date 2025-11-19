for f in */*.jsonl; do
  total=$(wc -l < "$f")
  correct=$(jq -r '.judge' "$f" | grep -c true)
  acc=$(awk "BEGIN {printf \"%.2f\", $correct / $total * 100}")
  echo "$f: $correct / $total = ${acc}%"
done

for f in */*/*.jsonl; do
  total=$(wc -l < "$f")
  correct=$(jq -r '.judge' "$f" | grep -c true)
  acc=$(awk "BEGIN {printf \"%.2f\", $correct / $total * 100}")
  echo "$f: $correct / $total = ${acc}%"
done


for f in *.jsonl; do
  total=$(wc -l < "$f")
  correct=$(jq -r '.judge' "$f" | grep -c true)
  acc=$(awk "BEGIN {printf \"%.2f\", $correct / $total * 100}")
  echo "$f: $correct / $total = ${acc}%"
done