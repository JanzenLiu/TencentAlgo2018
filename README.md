# TencentAlgo2018
A machine learning competition held by Tencent Social Ads to predict similarity between users: http://algo.qq.com/home/home/index.html

## Timeline
**Apr 18th 12:00:00 - May 9th 11:59:59:** Preliminaries A. 30% of the test data in the preliminaries stage will be used for the scoring and ranking. This stage will last for 3 weeks.

**May 9th 12:00:00 - May 23th 11:59:59:** Preliminaries B. The remained 70% of the test data in the preliminaries stage will be used for the scoring and ranking. The historical best score of each team in this stage will be used for the final ranking. Teams finally ranked top 20% (and at least top 200) will be selected to the playoff. This stage will last for 2 weeks.

**May 24th 12:00:00 - Jun 6th 11:59:59:** Playoff A. 30% of the test data in the playoff stage will be used for the scoring and ranking. This stage will last for 2 weeks.

**Jun 6th 12:00:00 - Jun 13th 11:59:59:** Playoff B. The remained 70% of the test data in the playoff stage will be used for the scoring and ranking. This stage will last for 1 weeks. The historical best score of each team in this stage will be used for the final ranking. Top 10 teams at the end will be selected to the defense. This stage will last for 1 week.

**Late June**: Defense and Awarding.

## Project Structure

    .
    ├── code                                # Source files
    │   ├── analysis                        # Analysis and visualization scripts
    │   ├── feature                         # Feature engineering scripts
    │   ├── model                           # Model scripts
    │   ├── pipeline                        # Pipeline scripts to run the whole (or partial) process
    │   ├── preprocess                      # Preprocessing scripts
    │   └── utils                           # Common tool and utility scripts
    ├── data                                # Data files
    │   └── raw                             # Raw data files downloaded from the competition
    │       └── preliminary_contest_data    # Data files for preliminary stage. You know where to download
    │           ├── adFeature.csv           # You know what it is. Just make sure the location is consistent
    │           ├── test1.csv               # Same as above
    │           ├── train.csv               # Same as above
    │           └── userFeature.data        # Same as above
    ├── docs                                # Documentation files
    ├── external                            # External utility files (e.g. external libraries, pretrained models and embeddings etc.)
    ├── figure                              # Figure files
    ├── log                                 # Log files
    ├── playground                          # Playground for members to place personal files (e.g. notebooks, scripts, outputs etc.)
    │   ├── Elvin                           # Elvin's personal files (you can rename it as you like)
    │   ├── Janzen                          # Janzen's personal files
    │   └── Lily                            # Lily's personal files (you can rename it as you like)
    ├── test                                # Automated tests
    └── README.md
