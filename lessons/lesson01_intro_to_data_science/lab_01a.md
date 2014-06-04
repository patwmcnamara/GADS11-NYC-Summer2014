# Computer Setup and Data Handling
## Macs: installing `homebrew`

`Homebrew` is a mac installer for packages/libraries/etc that works alongside Apple's installers. We need it for git. If typing `brew` into your terminal doesn't return an error, you're set. Otherwise:

```sh
ruby -e "$(curl -fsSL https://raw.github.com/mxcl/homebrew/go)"
```

If you recieve an SSL certificate error:

```sh
ruby -e "$(curl --insecure -fsSL https://raw.github.com/mxcl/homebrew/go)"
```

## `python`

While the class is about data science, the language we'll use to discuss it is Python. Python is the subject of the next lecture, so for now, you only need to get a useful distribution installed. The popular flavor these days is Anaconda. Installing is easy and will set up just about all the useful libraries that you'll want to use. <a href="https://store.continuum.io/cshop/anaconda/">Download and install here</a> for your computer.

**Note to Engineers:** If you prefer to not have anaconda's distribution as your primary python, comment out the `PATH` line for anaconda in `~/.bash_profile` and add an alias for anaconda's python, ipython and conda package handler:

```sh
alias apython="~/anaconda/bin/python"
alias ipython="~/anaconda/bin/ipython"
alias conda="~/anaconda/bin/conda"
```

For visualizations we'll primarily use matplotlib and yhat's version of ggplot for python:

```sh
conda install -c https://conda.binstar.org/public ggplot
```

Users experiencing ggplot package errors should try pip (this problem was observed on Ubuntu and Windows):

```sh
pip install ggplot
```

## Installing `git`
If typing `git` into your terminal doesn't return an error, you're set. Otherwise:

* Macs: `brew install git`
* Windows: Install git bash http://openhatch.org/missions/windows-setup/install-git-bash
    * The default options will probably work well for you
* Linux: If you're on linux you should already know how to do this with your package manager. On Ubuntu you can use `apt-get install git`, otherwise find your <a href="http://git-scm.com/download/linux">distribution</a>

**Note:** If you have issues with `brew install` because of an XCode error, try using this Heroku Toolbelt installation that will include git, or choose an OS based installation from this guide: http://git-scm.com/book/en/Getting-Started-Installing-Git

If you're unfamiliar with git, these tutorials are useful for learning the basic commands and for visualizing the notion of a branched git history:
https://try.github.io/levels/1/challenges/1
http://pcottle.github.io/learnGitBranching/

Remember: git is software for version management. github is an online service for hosting, sharing, and communicating around git repositories.

It's a good idea to do some basic git config:

```sh
git config --global user.name "Your Name"
git config --global user.email name@email.com
```

Once you've setup git and github, clone your fork of the class repository. We'll be using the <a href="https://help.github.com/articles/using-pull-requests#fork--pull">Fork and Pull git model</a>. If you understand the notion of branching, A Fork is like a branch that has a different owner, and Pull Requests are like asking to merge a branch from your fork into the master branch of the thing you forked. You have to make a request because you don't own it. You will be pushing changes to your forked repository, and submitting pull requests to the class repository.

From the github help page:
> The Fork & Pull Model lets anyone fork an existing repository and push changes to their personal fork without requiring access be granted to the source repository. The changes must then be pulled into the source repository by the project maintainer.

Begin by forking this repo into your own account. Now Github is hosting a copy of this repo under your username. Next you need to copy this onto your local machine so you can do work with it. This is called cloning:
```sh
cd ~/; git clone git@github.com:<your github username>/GADS11-NYC-Summer2014.git
```

for example:
```sh
cd ~/; git clone git@github.com:justalfred/GADS11-NYC-Summer2014.git
```

If this command fails, you can either use:
```sh
cd ~/; git clone https://github.com/<your github username>/GADS11-NYC-Summer2014.git
```

...but then you may have to authenticate every time you push or pull. Or you can [set up ssh forwarding](https://help.github.com/articles/generating-ssh-keys) and try again. It takes longer to set up, but then everything will just work in the future.

## Lab Submissions

We'll be using Pull Requests to submit assignments. As practice we'll walk you through submitting one now.

In `GADS11-NYC-Summer2014/lab_submissions/lab01` on your local machine, make a directory with your first initial/full last name.

```sh
DIR='flastname'; cd ~/GADS11-NYC-Summer2014/lab_submissions/lab01; mkdir $DIR; open $DIR
```

With a text or markdown editor, create and save a markdown file (`make_up_a_filename.md`) with the following content:

* Your name and what you do
* One liner about your coding and math background
* Any social web you use and don't mind sharing (twitter link, for example)
* A data blog post you read recently for sharing with the class

Create a branch of the repository with a unique name, and then commit to that repo. You will want to type `git status` between each command to see what changes have been made and what's in staging. You can also try `git log` before and after the following block to see updates to your history.

```sh
git checkout -b my_name_class_1        # create a new branch and enter it
git add .                              # add all changes to the stage
git commit -m 'my first git commit!'   # commit those changes into the history
git push origin my_name_class_1        # push (upload) the branch to origin
```

In the last command, `origin` refers to a remote. That is, an alias for a link to a repository. You can view a list of remotes and where they point using `git remote -v`.

Add a pull request. This is the actual submission of your work. You can do this on github by finding your branch and clicking "Create pull request." Developers, feel free to use some command line tool for this if you prefer it.

Again, a link to github documentation on the <a href="https://help.github.com/articles/using-pull-requests#fork--pull">Fork and Pull git model</a>.

As the root repo changes, you will periodically want to fetch those changes and merge them into your fork. First, you must set up a new remote that points to the original repo.

```sh
git remote add root git@github.com:patwmcnamara/GADS11-NYC-Summer2014.git
```

Next, you can fetch and merge in the single command:

```sh
git pull root master
```

Once your pull request has been accepted, you can run this command and the lastest version including your changes will now be in your master. At this point, you can get rid of your local branch if you want.

```sh
git checkout master                 # switch to branch master
git branch -d <your branch name>    # delete a branch
```

You can always type `git branch` to see what branches exist, and which one you're in. I don't know if it's necessary, but I always switch to `master` before pulling from `root/master`.

## Next Steps

We will always recommend 4 or 5 readings or other support materials for every class, either to supplement the current material, prep for the next class, or covering previous material that students still have questions on.

**Reading and other Materials**

* <a href="http://www.quora.com/Data-Science/What-is-it-like-to-be-a-data-scientist">Quora: What is it like to be a data scientist?</a>
* <a href="http://www.youtube.com/watch?v=h9vQIPfe2uU"> Josh Wills: The Life of a Data Scientist</a>
* <a href="http://www.p-value.info/"> P-Value.info, Carl Anderson's blog (Director of Data at Warby Parker)</a>
* <a href="http://blog.okcupid.com/"> A look at OKCupid and their detailed work in trends</a>
* <a href="http://radar.oreilly.com/2011/09/building-data-science-teams.html">DJ Patil, Building Data Science Teams</a>
* <a href="http://benfry.com/phd/">Ben Fry's Dissertation: Computational Information Design </a>
