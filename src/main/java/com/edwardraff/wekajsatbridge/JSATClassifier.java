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


import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;

/**
 * This wraps a JSAT classifier as a Weka Classifier 
 * 
 * @author Edward Raff
 */
public class JSATClassifier extends weka.classifiers.Classifier
{
    jsat.classifiers.Classifier classifier;

    /**
     * Creates a new Weka Classifier object that calls the given JSAT classifier
     * @param classifier the JSAT classifier to use
     */
    public JSATClassifier(jsat.classifiers.Classifier classifier)
    {
        this.classifier = classifier;
    }

    @Override
    public void buildClassifier(Instances data) throws Exception
    {
        ClassificationDataSet cds = (ClassificationDataSet) InstanceHandler.instancesToDataSet(data);
        classifier.trainC(cds);
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception
    {
        CategoricalResults results = classifier.classify(InstanceHandler.instanceToDataPoint(instance));
        //TODO should add support in JSAT to get the backing double array to avoid having to make a new one like this
        double[] dist = new double[results.size()];
        for(int i = 0; i < dist.length; i++)
            dist[i] = results.getProb(i);
        return dist;
    }

    @Override
    public Capabilities getCapabilities()
    {
        Capabilities capabilities = super.getCapabilities();
        //TODO a better way to do this? Not all JSAT methods will support both
        capabilities.enable(Capability.NOMINAL_ATTRIBUTES);
        capabilities.enable(Capability.NUMERIC_ATTRIBUTES);
        
        capabilities.enable(Capability.NOMINAL_CLASS);
        
        capabilities.setMinimumNumberInstances(1);
        return capabilities;
    }
    
    
}
