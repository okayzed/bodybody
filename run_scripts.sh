BAD_FILES=images/bad/*.*
GOOD_FILES=images/good/*.*
BLOG_FILES=blog/*.jpg

echo "Testing negatives"
python scripts/bodydetect.py ${BAD_FILES}

echo "Testing positives"
python scripts/bodydetect.py ${GOOD_FILES}

echo "Testing blog"
python scripts/bodydetect.py ${BLOG_FILES}
