
## Installation

After cloning the repository, run

> git submodule init
> git submodule update

To install the hugo theme

## Development

In debug, run website using

> hugo serve --disableFastRender -D

* -D to use drafts in posts folder without building to public folder
* --disableFastRender because it advised on LoveIt theme (https://hugoloveit.com/theme-documentation-basics/)

## Deployment

* set `draft: false` in header section of the post
* run

> hugo --minify

to build /docs folder (will be deployed to github pages)

* push to main

