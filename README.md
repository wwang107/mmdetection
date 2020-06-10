# Dockerize Human Detector

I add my implementation that extracts a bouding box of a detected human in an image and write it into a json file. The output path of the detected bouding box is currently fixed to my external harddrive (/media/weiwang/Elements/human36m). 

The script that execute the bouding box extraction is at "extractbbox.py"

# Visualizing the process in Visdom

Since the docker does not provide GUI, I use Visdom inside the docker. Run Visdom on port 8888 becausse port 8888 of the docker container has been mapped to port 8888 of the host machine. In this way, you can see visualization on the browser of host at localhost:8888

ssh -D 9999 -J iai cvg10
or
