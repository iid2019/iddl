# Git 使用指南

1. 在浏览器中打开[主项目地址](https://github.com/iid2019/iddl)，点击右上角的 fork 按钮，将该项目 fork 到自己的 GitHub 仓库中。

2. 在本地打开命令行，使用 `git clone` 命令将自己 fork 之后的仓库克隆到本地电脑当前路径下。

3. 本地当前阶段开发完成之后，使用 `git commit` 命令将改动提交到本地的仓库。

4. 在本地使用 `git push` 命令将本地的代码提交到自己的 GitHub 远程仓库。

5. 在浏览器中打开自己的项目地址，此时你 fork 的项目应会比上游的主项目提前了若干个提交，同时有可能落后了若干个提交。点击 pull request，发起新的 pull request。

6. 在经过代码审查之后，如果符合要求，该 pull request 中包含的 commits 会被合并到主项目中，这样你提交的代码就会添加到主项目中了；如果不符合要求，Derek 在相关的页面对不通过的原因进行陈述，请关闭该 pull request 并重新开发至符合要求，再发起新的 pull request。

7. 如果你的 pull request 已经被合并到主项目中，那么请务必在下一次开发之前，在本地执行 `git fetch upstream` 和 `git merge upstream/master` 命令，来达到与上游主项目同步的目的。第一次执行这一步时需要添加远程上游仓库: 执行 `git remote add upstream git@github.com:iid2019/iddl.git` 命令。

8. 继续进行下一阶段的开发，回到第 3 点。

9. 每次的 commit 必须加上一行清晰明了的 message 信息，尽量不要省略或填充没有意义的信息。

10. 文件命名皆采用小写英文字母 + 下划线连接，不要包含特殊字符。

> Notes
> - [关于与上游主项目同步](https://help.github.com/articles/syncing-a-fork/)
> - [关于 pull request](https://help.github.com/articles/about-pull-requests/)
> - [关闭 pull request](https://help.github.com/articles/closing-a-pull-request/)
