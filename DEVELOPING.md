## Installtion
Clone the repo and install the full dependencies
```
pip install -e ".[full]"
```

A clean conda environment is recommended for this. See [installation guide](./installation.rst) for more details.

## Git workflow
You should always use Git for making changes to the code. To enforce a unified code style automatically, `pre-commit` is used. A typical workflow should look something like this:
- create a new branch based off main, either using a GUI for git in your IDE of choice or through CLI `git checkout -b new_branch_name` (make sure you are at main by running `git checkout main` first before creating the new branch)
- make your changes
- run the tests to make sure nothing breaks (see below).
- Give a brief description of the changes in the CHANGELOG under dev version.
- stage the changes (GUI or `git add files_you_changed`). If you are using CLI and get lost, try `git status` to see what's going on.
- run `pre-commit`, or just `git commit "message you want to insert"`, or commit from GUI. `pre-commit` will run checks and files will be changed to conform to the *Black* code style if needed.
- If changes have been made, stage these changes and run `git commit -m "message you want to insert"` again.
- The commit messages **must follow** [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/). Otherwise it will be stopped by the pre-commit routine.
- Push to remote by running `git push -u origin new_branch_name`. If you have already pushed it before then simply `git push`.
- Go to the github repo and create a pull request. **MAKE SURE** you ask for a review and have it approved with all tests passed. Branch protection rules may not be in place and in no circumstance should you merge into main without asking.
- After pull request is approved and merged, delete the branch on github.
- In your local repo, switch to main and then run `git pull`. Delete your local repo if you want.

## Tests
Whenever you have made some changes to the code, you want to make sure the code still passes all the tests, and add new tests or change the old ones to cover the additions.

In the top level of the source directory, run:
```
pytest --cov-config=.coveragerc --cov=meer21cm tests/
```

The above command run the tests and generate a report. Make sure there are no failed tests before you push.

You may want to add tests to cover more lines of code. There are multiple output formats available to check which lines are getting covered. I find html to be the easiest to read locally:
```
pytest --cov-config=.coveragerc --cov=meer21cm tests/ --cov-report term --cov-report html:coverage.html
```
This will generate a folder `coverage.html` (do not commit it, leave it untracked). You can open the html files inside to see the coverage.

When you push, Github Actions have been set up so the tests will be checked and a coverage report will also be generated on codecov (see [here](https://app.codecov.io/gh/meerklass/meer21cm/pull/130) for an example).You should always first check locally that the tests are good before pushing to a PR to trigger the tests on Github.

## Documentation
When you write new functions and classes, they should be properly documented with docstrings. You can check how they will look in the documentation website by creating the website locally. In the `meer21cm` directory:

```
cd docs
mkdir build
make html
```

The HTML pages should be created which you can then open in your browser. If you have made some changes you can always update the htmls

```
make clean
make html
```


## Caution
- Avoid tracking a large file. **Never commit** a change where **a large file** (>50MB) is added to the repo. This should be prohibited by `pre-commit` anyway, but just in case it is not set up properly you should be aware as well. If you have done so, you need to rewind back to the commit before that and start over. No, deleting it in a later commit will not fix it.
- **Never merge into `main` locally**. The only way `main` can be changed is through pull request and pull from remote.
- Try not to break API, although it will happen since we are at early stages. For example, if the original code has a function `def func1(arg1=None)` and you changed it to `def func1(arg_1=None)`, it breaks API. That is because scripts using older versions of the code will stop working since `func1(arg1=something)` will return an error. `def func1(arg1=None,arg2=None)` is not breaking API for example, because `func1(arg1=something)` still works.
- **Never let AI write unit tests**. The copilot/cursor AI autocomplete stuff is very useful (so are the agents), but one can easily become lazy and approve AI-generated changes that may be wrong. Avoid AI-designed code in unit test to make sure errors are caught.

## Recommended good practice
These are some practices that are standard in development. They are not enforced (and admittedly I don't follow them all the time), but you should be aware of them.

- Provide a synopsis of your commits in the `CHANGELOG`
- Follow [semantic versioning](https://semver.org/) and update the version number of the code appropriately (this will be enforced later after release of v1)
