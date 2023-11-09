
BRANCH=fix_inference_scripts
git init
git checkout -b $BRANCH
git add .
git commit -F commit_info
git push origin $BRANCH
