i='hallo'
j='hallo'
if [ $i==$j ]; then
    echo "I'm right."
else
    echo "I'm wrong."
fi

echo "${i%o}"