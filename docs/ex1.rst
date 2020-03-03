Exercise 1 - Understanding The Pumpkin
______________________________________

If a person is told to count pumpkins on an image, that is all the information they need.
We have prior knowledge about what pumpkins look like and can use that information to distinguish between them and other objects or the background.
A computer does not have that knowledge, therefor it needs a definition of pumpkins.

On the input image we have orange pumpkins on a green-brown background.
Using this fact, one of the easiest ways to utilize computer vision for separating the pumpkins from the background is by color information.
To learn pumpkin colors, we are annotating pumpkins on the image by hand, removing everything else from the image, and checking the three-channel histograms for the annotated parts.
The goal is to know the color mean and standard deviation of pumpkins.
