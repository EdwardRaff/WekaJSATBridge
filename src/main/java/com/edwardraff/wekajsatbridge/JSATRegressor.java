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


import jsat.regression.RegressionDataSet;
import jsat.regression.Regressor;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;

/**
 * This wraps a JSAT Regressor as a Weka 'Classifier' that works on regression tasks
 *
 * @author Edward Raff
 */
public class JSATRegressor extends weka.classifiers.Classifier
{
    Regressor regressor;

    public JSATRegressor(Regressor regressor)
    {
        this.regressor = regressor;
    }

    @Override
    public void buildClassifier(Instances data) throws Exception
    {
        RegressionDataSet rds = (RegressionDataSet) InstanceHandler.instancesToDataSet(data);
        regressor.train(rds);
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception
    {
        return regressor.regress(InstanceHandler.instanceToDataPoint(instance));
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
