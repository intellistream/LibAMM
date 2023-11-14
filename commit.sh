

BRANCH=fix_srand_bugs_ds

git init
git checkout -b $BRANCH
git add .
git commit -F commit_info
git push origin $BRANCH
