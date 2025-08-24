package empmanager;

import javafx.beans.property.SimpleDoubleProperty;
import javafx.beans.property.SimpleStringProperty;
import javafx.beans.property.SimpleIntegerProperty;


/**
 *
 * @author ballaj
 */



public class Employee {
    
    /**Create private properties */
    
    private final SimpleIntegerProperty id;
    private final SimpleStringProperty firstName;
    private final SimpleStringProperty lastName; 
    private final SimpleStringProperty middleName;
    private final SimpleStringProperty position;
    private final SimpleDoubleProperty salary;
    
    
    /**Constructor with specific properties */
    
    public Employee (int id, String firstName, String lastName, String middleName, String position, double salary)
    {
        this.id=new SimpleIntegerProperty(id);
        this.firstName=new SimpleStringProperty(firstName);
        this.lastName= new SimpleStringProperty(lastName);
        this.middleName=new SimpleStringProperty(middleName);
        this.position=new SimpleStringProperty(position);
        this.salary=new SimpleDoubleProperty(salary);
    }
    
    /**getter and setter */
    public int getId()
    {
        return id.get();
    }
    
    public String getFirstName()
    {
        return firstName.get();
    }
    
    public String getMiddleName()
    {
        return middleName.get();
    }
    
    public String getLastName()
    {
        return lastName.get();
    }
    
    public String getPosition()
    {
        return position.get();
    }
    
    public Double getSalary()
    {
        return salary.get();
    }
    
    
    public void setId(int id)
    {
        this.id.set(id);
    }
    
    public void setFirstName(String firstName)
    {
        this.firstName.set(firstName);
    }
    
    public void setMiddleName(String middleName)
    {
        this.middleName.set(middleName);
    }
    
    public void setLastName(String lastName)
    {
        this.lastName.set(lastName);
    }
    
    public void setPosition(String position)
    {
        this.position.set(position);
    }
    
    public void setSalary(double salary)
    {
        this.salary.set(salary);
    }
}
