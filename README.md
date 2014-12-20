WekaJSATBridge
==============

A mini library to convert between JSAT and Weka datasets/objects, allowing some Weka objects to work in JSAT and vice versa

The intention is to provide wrappers that implement the interface/API for one package (say JSAT) by calling to an object from the other (say Weka). By implementing thise with utilities to convert between JSAT and Weka DataSets/Instances, most of the functionality can be shared with relative ease. 

Currently has wrappers for:
  * Weka -> JSAT
    * Classification
    * Regression
    * Clustering
  * JSAT -> Weka 
    * Classification
    * Regression
