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

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author Edward Raff
 */
public class OtherUtils
{
    public static <T> T serializationCopy(T toCopy)
    {
        try(ByteArrayOutputStream bout = new ByteArrayOutputStream();
                ObjectOutputStream oout = new ObjectOutputStream(bout))
        {
            oout.writeObject(toCopy);
            oout.flush();
            try(ByteArrayInputStream bin = new ByteArrayInputStream(bout.toByteArray());
                    ObjectInputStream oin = new ObjectInputStream(bin))
            {
                return (T) oin.readObject();
            }
            catch (ClassNotFoundException ex)
            {
                //shouldn't happen, the class is the one you gave us!
                Logger.getLogger(OtherUtils.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
        catch(IOException ex)
        {
            //shouldn't happen, we are using a byteArray
            Logger.getLogger(OtherUtils.class.getName()).log(Level.SEVERE, null, ex);
        }
        return null;
    }
}
