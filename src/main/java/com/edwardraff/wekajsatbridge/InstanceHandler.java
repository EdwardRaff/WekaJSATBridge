package com.edwardraff.wekajsatbridge;

/*
 * Copyright (C) 2014 Edward Raff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */


import jsat.DataSet;
import jsat.SimpleDataSet;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.DataPoint;
import jsat.linear.DenseVector;
import jsat.linear.IndexValue;
import jsat.linear.SparseVector;
import jsat.linear.Vec;
import jsat.regression.RegressionDataSet;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;
import weka.core.UnassignedDatasetException;

/**
 * This class provides methods to convert between JSAT and Weka datasets and 
 * instances in both directions (JSAT to Weka and Weka to JSAT)
 * 
 * @author Edward Raff
 */
public class InstanceHandler
{

    /**
     * Attempts to convert a Weka Instance object into a JSAT DataPoint.<br>
     * Note, that JSAT dosn't support all of the possible types of an Instance,
     * such as strings, and those will be ignored. It is also possible for
     * accessing the values of an arbitrary instance to throw an exception.
     *
     * @param instance the instance to convert.
     * @return a new DataPoint object representing the instance
     */
    public static DataPoint instanceToDataPoint(Instance instance)
    {
        int numAttributes = instance.numAttributes();

        int numNumeric = 0;
        int numNominal = 0;

        //we need to figure out if this instance has the class value in it, which can throw an exception
        int classIndex = -1;//negative means not present
        try
        {
            classIndex = instance.classIndex();
        }
        catch (UnassignedDatasetException ex)
        {
            classIndex = -1;//nodataset for instance
        }

        for (int i = 0; i < numAttributes; i++)
            if (i == classIndex)
                continue;
            else if (instance.attribute(i).isNumeric())
                numNumeric++;
            else if (instance.attribute(i).isNominal())
                numNominal++;

        //now we can create the data point
        int[] nominalValues = new int[numNominal];
        CategoricalData[] catInfo = new CategoricalData[numNominal];
        Vec numericValues;
        if (instance instanceof SparseInstance)
            numericValues = new SparseVector(numNumeric);//TODO would be faster to allocate and fill the int/double arrays ourselves
        else
            numericValues = new DenseVector(numNumeric);
        //fill values
        int numericPos = 0, nominalPos = 0;
        for (int i = 0; i < numAttributes; i++)
        {
            Attribute attribute_i = instance.attribute(i);
            if (attribute_i.isNumeric())
            {
                numericValues.set(numericPos++, instance.value(i));
            }
            else if (attribute_i.isNominal())
            {
                catInfo[nominalPos] = new CategoricalData(attribute_i.numValues());
                nominalValues[nominalPos++] = (int) instance.value(i);
            }
        }

        return new DataPoint(numericValues, nominalValues, catInfo, instance.weight());
    }

    /**
     * Attempts to convert the given set of Instances into a JSAT dataset. Based
     * on the class attribute of the instances, the returned DataSet may be a 
     * {@link SimpleDataSet}, {@link RegressionDataSet}, or 
     * {@link ClassificationDataSet}. 
     * @param instances the Weka style dataset to convert to a JSAT one
     * @return the appropriate JSAT dataset type for the given data
     */
    public static DataSet instancesToDataSet(Instances instances)
    {
        int numAttributes = instances.numAttributes();
        
        int numNumeric = 0;
        int numNominal = 0;
        
        //we need to figure out if this instance has the class value in it, which can throw an exception
        
        int classIndex = -1;//negative means not present
        classIndex = instances.classIndex();//dosn't throw an exception when its a Instances object
        
        int nominalPos = 0, numericPos = 0;
        
        for(int i = 0; i < numAttributes; i++)
            if(i == classIndex)
                continue;
            else if(instances.attribute(i).isNumeric())
                numNumeric++;
            else if(instances.attribute(i).isNominal())
                numNominal++;
        
        CategoricalData[] catInfo = new CategoricalData[numNominal];
        //go back and get nominal info
         for(int i = 0; i < numAttributes; i++)
            if(i == classIndex)
                continue;
            else if(instances.attribute(i).isNominal())
                catInfo[nominalPos++] = new CategoricalData(instances.attribute(i).numValues());
         
        
        
        DataSet dataSet;
        if(classIndex < 0)//no target value
            dataSet = new SimpleDataSet(catInfo, numNumeric);
        else//classification or regression?
        {
            Attribute classAttribute = instances.classAttribute();
            if(classAttribute.isNumeric())//regression
                dataSet = new RegressionDataSet(numericPos, catInfo);
            else if(classAttribute.isNominal())//classificaiton
                dataSet = new ClassificationDataSet(numericPos, catInfo, new CategoricalData(classAttribute.numValues()));
            else
                throw new RuntimeException("Class attribute is not a numeric or nominal value");
        }
        
        for(int i = 0; i < instances.numAttributes(); i++)
        {
            Instance instance = instances.instance(i);
            int[] nominalVals = new int[numNominal];
            Vec numericVals;
            if(instance instanceof SparseInstance)
                numericVals = new SparseVector(numNumeric);
            else
                numericVals = new DenseVector(numNumeric);

            numericPos = 0;
            nominalPos = 0;
            for (int j = 0; j < instances.numAttributes(); j++)
            {
                if(j == classIndex)
                    continue;
                Attribute attribute_i = instance.attribute(j);
                if (attribute_i.isNumeric())
                    numericVals.set(numericPos++, instance.value(j));
                else if (attribute_i.isNominal())
                    nominalVals[nominalPos++] = (int) instance.value(j);
            }

            DataPoint dp = new DataPoint(numericVals, nominalVals, catInfo, instance.weight());
            //add to the dataset 
            if(dataSet instanceof RegressionDataSet)
                ((RegressionDataSet)dataSet).addDataPoint(dp, instance.value(classIndex));
            else if(dataSet instanceof ClassificationDataSet)
                ((ClassificationDataSet)dataSet).addDataPoint(dp, (int) instance.value(classIndex));
            else//just a dataset
                ((SimpleDataSet)dataSet).getBackingList().add(dp);
        }
        
        return dataSet;
    }
    
    /**
     * Converts a JSAT DataPoint to a Weka Instance object
     * @param dp the datapoint to convert to a Weka Instance
     * @return the Weka Instance representing this DataPoint
     */
    public static Instance dataPointToInstance(DataPoint dp)
    {
        int[] nominalValues = dp.getCategoricalValues();
        Vec numericValues = dp.getNumericalValues();

        Instance instance = new Instance(nominalValues.length + numericValues.length());
        int pos = 0;
        for (int i = 0; i < nominalValues.length; i++)
            instance.setValue(pos++, nominalValues[i]);
        for (int i = 0; i < numericValues.length(); i++)
            instance.setValue(pos++, numericValues.get(i));
        instance.setWeight(dp.getWeight());
        return instance;
    }
    
    /**
     * Converts a JSAT dataset into a Weka Instances object with the instance 
     * already in it. If the dataSet is a {@link ClassificationDataSet} or 
     * {@link RegressionDataSet} the Instances object will have a class index, 
     * and the class index will always be the last index. 
     * 
     * @param dataSet the dataset to convert to a Weka dataset
     * @return the Weka Instances object version of this JSAT dataset
     */
    public static Instances dataSetToInstances(DataSet dataSet)
    {
        FastVector attributes = new FastVector();
        
        CategoricalData[] catInfo = dataSet.getCategories();
        for(int i = 0; i < catInfo.length; i++)
        {
            CategoricalData cat = catInfo[i];
            String name = cat.getCategoryName()+i/*make sure they are different incase of "No Name"*/;
            attributes.addElement(categoricalDataToAttribute(cat, name));
        }
        
        for(int i = 0; i < dataSet.getNumNumericalVars(); i++)
            attributes.addElement(new Attribute("numericAtt"+i));
        
        
        //class attribute?
        int classIndex = -1;
        if(dataSet instanceof RegressionDataSet)
        {
            classIndex = attributes.size();
            attributes.addElement(new Attribute("regressionTarget"));
        }
        else if(dataSet instanceof ClassificationDataSet)
        {
            classIndex = attributes.size();
            attributes.addElement(categoricalDataToAttribute(((ClassificationDataSet)dataSet).getPredicting(), "classTarget"));
        }
        
        Instances instances = new Instances("JSATtoWekaDataset", attributes, dataSet.getSampleSize());
        
        instances.setClassIndex(classIndex);
        
        for(int i = 0; i < dataSet.getSampleSize(); i++)
        {
            DataPoint dp =  dataSet.getDataPoint(i);
            double targetValue = Double.NaN;
            if(dataSet instanceof RegressionDataSet)
                targetValue = ((RegressionDataSet)dataSet).getTargetValue(i);
            else if(dataSet instanceof ClassificationDataSet)
                targetValue = ((ClassificationDataSet)dataSet).getDataPointCategory(i);
            
            //TODO handle sparse data points
            double[] attValues = new double[attributes.size()];
            int pos = 0;
            for(int catVal : dp.getCategoricalValues())
                attValues[pos++] = catVal;
            for(IndexValue iv : dp.getNumericalValues())
                attValues[pos++] = iv.getValue();
            if(classIndex >= 0)
                attValues[classIndex] = targetValue;
            Instance instance = new Instance(dp.getWeight(), attValues);
            instance.setDataset(instances);//is this needed?
            instances.add(instance);
        }
        
        return instances;
    }

    /**
     * Helper method that converts a CategoricalData object into a Weka 
     * Attribute object
     * @param cat the categoricaldata object to convert
     * @param name the name to use for the Attribute's name
     * @return a Weka Attribute object representing the same nominal variable
     */
    private static Attribute categoricalDataToAttribute(CategoricalData cat, String name)
    {
        FastVector attributeValues = new FastVector(cat.getNumOfCategories());
        for(int j = 0; j < cat.getNumOfCategories(); j++)
            attributeValues.addElement(cat.getOptionName(j));
        Attribute catAtt = new Attribute(name, attributeValues);
        return catAtt;
    }
    
    /**
     * Helper method that converts a CategoricalData object into a Weka 
     * Attribute object with the same name. 
     * @param cat the categoricaldata object to convert
     * @return a Weka Attribute object representing the same nominal variable
     */
    private static Attribute categoricalDataToAttribute(CategoricalData cat)
    {
        return categoricalDataToAttribute(cat, cat.getCategoryName());
    }
}
