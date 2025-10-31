## How to contribute
We'd love to accept your patches and contributions to this project. There are just a few small guidelines you need to follow.

### Submitting a patch

  1. It's generally best to start by opening a new issue describing the bug or
     feature you're intending to fix.  Even if you think it's relatively minor,
     it's helpful to know what people are working on.  Mention in the initial
     issue that you are planning to work on that bug or feature so that it can
     be assigned to you.

  2. Follow the normal process of [forking][] the project, and setup a new
     branch to work in.  It's important that each group of changes be done in
     separate branches in order to ensure that a pull request only includes the
     commits related to that bug or feature.

  3. To ensure properly formatted code, please use [Ruff][] as linter and code
     formatter. Ruff is fast and replaces many other tools. Run `ruff check`
     on the files you edited (or globally). Also, if you are using VSCode,
     there is a Ruff extension that will make the formatting and linting
     automatically for you (make sure you have `format-on-save` option enabled
     in your VSCode). It's necessary that your code is completely
     "lint-free", so Ruff will help you find and fix common style issues.

  1. Any significant changes should almost always be accompanied by tests. The
     project already has good test coverage, so look at some of the existing
     tests if you're unsure how to go about it. We're using [coveralls][] that
     is an invaluable tools for seeing which parts of your code aren't being
     exercised by your tests.

  2. Do your best to have [well-formed commit messages][] for each change.
     This provides consistency throughout the project, and ensures that commit
     messages are able to be formatted properly by various git tools.

  3. Finally, push the commits to your fork and submit a [pull request][]. Please,
     remember to rebase properly in order to maintain a clean, linear git history.

[forking]: https://help.github.com/articles/fork-a-repo
[ruff]: https://docs.astral.sh/ruff/
[coveralls]: https://coveralls.io
[well-formed commit messages]: https://www.conventionalcommits.org/en/v1.0.0/
[pull request]: https://help.github.com/articles/creating-a-pull-request
