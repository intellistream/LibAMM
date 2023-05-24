BRANCH=REFINE_SCAN
git init
git checkout -b $BRANCH
git add .
git commit -F commit_info
git push origin $BRANCH
