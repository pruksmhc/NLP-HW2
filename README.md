
My game plan:

-I will need to change the DataLoader to output 2 TENSORS (2 sentences)
-You pad it separately.  

yoU don't hVE TO 1-HOT ENCODE BECAUSE the RNN will do that for you. []
I will download it to the RNN.. 

We sort the dataset in the collate function. 