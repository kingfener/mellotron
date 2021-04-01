


echo "s1=$1"
echo "s2=$2"
git add $1 
git commit -m $2 
git push
git subtree push --prefix=taco_sma taco_sma master







