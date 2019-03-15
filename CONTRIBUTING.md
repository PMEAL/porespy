# Contributing to *PoreSpy*

PoreSpy is a collection of image analysis functions we use regularly in our research.  We hope that other researchers will contribute their own functions to this package, to help make it even more useful.  This guide is intended to make it as easy as possible to get involved.

Before you start you'll need to set up a free GitHub account and sign in. Here are some [instructions][link_signupinstructions] to get started.

## Ways to Contribute

### Open a New Issue

We use Github to track [issues][link_issues].  Issues can take the form of

(a) bug reports such as a function producing an error or odd result in some circumstances.

(b) feature requests such a suggest a new function be added to the package, presumably based on some literature report that describes it.  Alternatively, this may mean enhancements to an existing function.

(c) general usage questions where the documentation is not clear and the you need help getting a function to work as desired.  This is actually a bug report in disguise since it means there is a problem with the documentation.

### Addressing Open Issues

Help fixing open [issues][link_issues] is always welcome; however, the learning curve for submitting new code to any repo on Github is a bit intimidating.  The process is as follows:

a) [Fork][link_fork] PoreSpy to your own account. This lets you work on the code since you know own that forked copy.

b) Pull the code to your local machine using some Git client.  We suggest [GitKraken][link_gitkraken].

c) Create a new branch, with a useful name like "fix_issues_011" or "add_awesome_filter", then checkout that branch.  For help using the Git version control system, try [these resources][link_using_git].

d) Edit the code as desired, either fixing or adding something.  You'll need to know Python and the various packages in the [Scipy][link_scipy] stack for this part.  

e) Push the changes back to Github, to your own repo.

f) Navigate to the the [pull requests area][link_pull_requests] on the PoreSpy repo, then click the "new pull request" button.  As the name suggests, you are asking us to [pull][link_pullrequest] your code in to our repo.  You'll want to select the correct branch on your repo (e.g. "add_awesome_filter") and the "dev" branch on PoreSpy.

g) This will trigger several things on our repo, including most importantly a conversation between you and the PoreSpy team about your code.  After any fine-tuning is done, we will merge your code into PoreSpy, and your contribution will be immortalized in PoreSpy.  


[link_github]: https://github.com/
[link_issues]: https://github.com/PMEAL/porespy/issues
[link_gitkraken]: https://www.gitkraken.com/
[link_pull_requests]: https://github.com/PMEAL/porespy/pulls
[link_fork]: https://help.github.com/articles/fork-a-repo/
[link_signupinstructions]: https://help.github.com/articles/signing-up-for-a-new-github-account
[link_pullrequest]: https://help.github.com/articles/creating-a-pull-request/

[link_using_git]: http://try.github.io/

[link_scipy]: https://www.scipy.org/
