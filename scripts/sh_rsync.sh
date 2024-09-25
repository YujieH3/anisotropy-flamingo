#!/bin/bash

src="/cosma8/data/do012/dc-he4/mock_lightcones/"
dest="/cosma8/data/do012/dc-he4/mock_lightcones_copy/"
rsync -av --update $src $dest