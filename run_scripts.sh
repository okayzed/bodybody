BAD_FILES=images/bad/*.*
GOOD_FILES=images/good/*.*

echo "Testing negatives"
python scripts/bodydetect.py ${BAD_FILES}

echo "Testing positives"
python scripts/bodydetect.py ${GOOD_FILES}
