# BKFZ-Demo
This is a minimal MNIST Tensorflow Demo for the BKFZ Summer School with the PHT-meDIC.

Analyst should upload the algorithm `analysis-train.py` as entrypoint in the central UI.
Admins should upload the corresponding data to their station.

If you are in the second round in thegr group Admins, you dont need to upload the data to the station again!

## Analysts
The analyst instructions you can find [in this link](https://docs.google.com/document/d/1GjOQORNVARs0uV3hLOmFzA8CiVO30EC6SNasrc5lA5M/edit?usp=sharing)

### Analyses code
This code loads a prepared dataset and partial trains a MNIST demo. Furthermore it uses Paillier encryption to securely count the number of training and testing samples (globally and privacy preserving)


### Cheat sheet
1. Download the App
2. Create and upload both keys
3. Submit an analysis
4. Download and decrypt results 

Please ask for help if you have in any of those steps issues.


## Admins

Your instructions you can find [in this link](https://docs.google.com/document/d/1OMPSwJ8r1PdFCvYKmxxxWzE4pVJ99mBsiuzsCJZGZ0s/edit?usp=sharing)

### Cheat sheet
1. Create keys
2. Login to the VM
3. Install a PHT-meDIC station
4. Approve and run analysis

Please ask for help if you have in any of those steps issues.


To copy demo train data to a station (replace YOUR_KEY, YOUR_DATA_PATH, STATION_NUM, IP_VM) and run this command:
``` shell
scp -i ~/.ssh/YOUR_KEY -r /YOUR_DATA_PATH/ds_STATION_NUM ubuntu@IP_VM:/home/ubuntu/station/station_data/
```


To run the train use this command (replace STATION_NUM and TRAIN_ID):
```json
{"repository":
    "dev-harbor.personalhealthtrain.de/demo-station-STATION_NUM/TRAIN_ID",
    "tag": "latest",
    "volumes":
        {"/home/ubuntu/station/station_data/ds_STATION_NUM":
        {"bind":
            "/opt/train_data/ds_STATION_NUM",
            "mode": "ro"}},
    "gpus":"all"
}
```


