package empmanager;
/**
 * Application name: Employee manager.
 * Author: Yassine Ballaj.
 * Started Date: 3/23/2015.
 * finished Date: 
 */


import javafx.application.Application;

import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import java.sql.*;
import javafx.scene.input.KeyCode;
import javafx.scene.Scene;
import java.util.ArrayList;
import javafx.geometry.Insets;
import javafx.scene.control.Button;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.scene.text.Font;
import javafx.scene.text.Text;
import javafx.stage.Modality;

import javafx.stage.Stage;

/**
 *
 * @author ballaj
 */
public class EmployeeManager extends Application {
    
    private SearchPane searchPane =new SearchPane();
    private Connection conn;
    private Statement stm;
    private int [] startingPoint=new int []{0};
    private ArrayList<Employee> employees =new ArrayList<Employee>();
    
    
    
    @Override
    public void start(Stage primaryStage) {
        
        /**Connect to a database  */
        connectToDataBase();
        
        listAllEmployee();
     
        employees.addAll(searchPane.data);
        
        /**List all Employee in the database */
         searchPane.listAllButton.setOnAction(new EventHandler<ActionEvent>() {

             public void handle(ActionEvent event) {
                 
                 searchPane.data.clear();
                 searchPane.data.addAll(employees);
                  
             }
         });
         
         
         /**Add Employee  */
         searchPane.addEmployee.setOnAction(new EventHandler<ActionEvent>(){
                 public void handle (ActionEvent event)
                 {
                     new AddEmployee(primaryStage,stm, searchPane.data, startingPoint,employees);
                 }
         });
         
         
         
         /**Delete employee */
         searchPane.deleteEmployee.setOnAction(e->{
             new DeleteEmployee(primaryStage,stm,searchPane.data,employees);
           
         });
           
         
          
         
         /**Update Employee */
         searchPane.updateEmployee.setOnAction(e->{new UpdateEmployee(primaryStage, stm, searchPane.data, startingPoint,employees);});
         
         
         
         /**Search */
         searchPane.setOnKeyPressed(e->{
         
            if(e.getCode()==KeyCode.ENTER)
            {
                search(primaryStage);
            }
             
         });
        
         
         /**Exit the application*/
        searchPane.exit.setOnAction(new EventHandler<ActionEvent>(){
           public void handle(ActionEvent event)
           {
               System.exit(0);
           }
        
        });
         
         
        /**Create a scene  */
        Scene scene =new Scene(searchPane,610,500);
        primaryStage.setScene(scene);
        primaryStage.setTitle("Employee Manager");
        primaryStage.show();
        
    }
    
    
    

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        launch(args);
    }
    
    
    
   
    public void connectToDataBase()
    {
        try {
            
           Class.forName("com.mysql.jdbc.Driver");
            
            conn=DriverManager.getConnection("jdbc:mysql://localhost:3306/EmployeeManager","root","");
            
            stm=conn.createStatement();
            
        }catch(Exception exception)
        {
            System.out.println(exception.getMessage());
        }
    }
    
    
    
    
    
    
    public void listAllEmployee()
    {
        
        try{
            
               String query ="select * from employee where id>'"+startingPoint[0]+"'";
               ResultSet resultSet=stm.executeQuery(query);
               
               while(resultSet.next())
               {
                    searchPane.data.add(new Employee(
                    Integer.parseInt(resultSet.getString(1)),
                    resultSet.getString(2),
                    resultSet.getString(4),
                    resultSet.getString(3),
                    resultSet.getString(5),
                    resultSet.getDouble(6)
                   ));
                   
                  startingPoint[0]=Integer.parseInt(resultSet.getString(1));
               }
               
            
        }catch(Exception exception)
        {
            System.out.println(exception.getMessage());
        }
       
    }
    
    
    
    
    /**Search method */
    public void search(Stage primaryStage)
    {
        int index=-1;
        int exist=0;
        ArrayList<Integer> indexes=new ArrayList<>();
        
        if (searchPane.searchField.getText().equals(""))
            {
                searchPane.searchField.requestFocus();
            }
        else
        {
        
        if (searchPane.comboBoxSearch.getValue().equals("Id"))
        {
                
                for(int i=0; i<employees.size();i++)
                {
                    
                    if (employees.get(i).getId()==Integer.parseInt(searchPane.searchField.getText()))
                    {
                       exist=1;
                       index=i;
                    }
                }
                if (exist==1)
                {
                   searchPane.data.clear();
                   searchPane.data.add(employees.get(index));
                   startingPoint[0]=0;
                }
                else
                {
                    popErrorMessage(primaryStage,"Employee Does Not Exist !!!");
                }
          }
        else if (searchPane.comboBoxSearch.getValue().equals("First name"))
        {
                
                for(int i=0; i<employees.size();i++)
                {
                    
                    if (employees.get(i).getFirstName().equals(searchPane.searchField.getText()))
                    {
                       exist=1;
                       indexes.add(i);
                    }
                }
                if (exist==1)
                {
                   searchPane.data.clear();
                   for(int i=0; i<indexes.size();i++) 
                   {
                      searchPane.data.add(employees.get(indexes.get(i)));
                   }
                   
                   startingPoint[0]=0;
                }
                else
                {
                    popErrorMessage(primaryStage,"Employee Does Not Exist!!!");
                }
        }
         else if (searchPane.comboBoxSearch.getValue().equals("Middle name"))
        {
                
                for(int i=0; i<employees.size();i++)
                {
                    
                    if (employees.get(i).getMiddleName().equals(searchPane.searchField.getText()))
                    {
                       exist=1;
                       indexes.add(i);
                    }
                }
                if (exist==1)
                {
                   searchPane.data.clear();
                   for(int i=0; i<indexes.size();i++) 
                   {
                      searchPane.data.add(employees.get(indexes.get(i)));
                   }
                   
                   startingPoint[0]=0;
                }
                else
                {
                    popErrorMessage(primaryStage,"Employee Does Not Exist!!!");
                }
          }
        
         else if (searchPane.comboBoxSearch.getValue().equals("Last name"))
        {
                
                for(int i=0; i<employees.size();i++)
                {
                    
                    if (employees.get(i).getLastName().equals(searchPane.searchField.getText()))
                    {
                       exist=1;
                       indexes.add(i);
                    }
                }
                if (exist==1)
                {
                   searchPane.data.clear();
                   for(int i=0; i<indexes.size();i++) 
                   {
                      searchPane.data.add(employees.get(indexes.get(i)));
                   }
                   
                   startingPoint[0]=0;
                }
                else
                {
                    popErrorMessage(primaryStage,"Employee Does Not Exist!!!");
                }
          }
         else if (searchPane.comboBoxSearch.getValue().equals("Position"))
        {
                
                for(int i=0; i<employees.size();i++)
                {
                    
                    if (employees.get(i).getPosition().equals(searchPane.searchField.getText()))
                    {
                       exist=1;
                       indexes.add(i);
                    }
                }
                if (exist==1)
                {
                   searchPane.data.clear();
                   for(int i=0; i<indexes.size();i++) 
                   {
                      searchPane.data.add(employees.get(indexes.get(i)));
                   }
                   
                   startingPoint[0]=0;
                }
                else
                {
                    popErrorMessage(primaryStage,"Employee Does Not Exist!!!");
                }
        }
         else if (searchPane.comboBoxSearch.getValue().equals("Salary"))
        {
                
                for(int i=0; i<employees.size();i++)
                {
                    
                    if (employees.get(i).getSalary()==Double.parseDouble(searchPane.searchField.getText()))
                    {
                       exist=1;
                       indexes.add(i);
                    }
                }
                if (exist==1)
                {
                   searchPane.data.clear();
                   for(int i=0; i<indexes.size();i++) 
                   {
                      searchPane.data.add(employees.get(indexes.get(i)));
                   }
                   
                   startingPoint[0]=0;
                }
                else
                {
                    
                    popErrorMessage(primaryStage,"Employee Does Not Exist!!!");
                }
          }
       }
    }
    
    
    
    
    /**Error Message */
    public void popErrorMessage(Stage primaryStage,String text2)
        {
            /**Create a Stage  */
            final Stage dialog=new Stage ();
            dialog.initModality(Modality.APPLICATION_MODAL);
            dialog.initOwner(primaryStage);
            
            /**Set text properties */
            Text text=new Text();
            text.setText(text2);
            text.setFill(Color.RED);
            text.setFont(Font.font("Times new roman",20));
            
            /**Set image properties */
            ImageView image =new ImageView(new Image("file:warning.png"));
           
             /**Set the button properties */
            Button dialogOk=new Button("OK");
            dialogOk.setPrefWidth(100);
            
            /**Set dialog HBox properties */
            HBox dialogHbox=new HBox ();
            dialogHbox.setSpacing(10);
            dialogHbox.getChildren().addAll(image,text);
            HBox.setMargin(image,new Insets(0,0,0,10));
            /**Set VBox properties */
            VBox dialogVBox =new VBox();
            dialogVBox.setSpacing(10);
            dialogVBox.getChildren().addAll(dialogHbox,dialogOk);
            VBox.setMargin(dialogOk,new Insets(0,0,0,170));
            
           
            
            /** exit the pop up window*/
            dialogOk.setOnAction(e->{dialog.close();});
            
            /**Create a scene  */
            Scene scene=new Scene (dialogVBox,450,100);
            
            dialog.setScene(scene);
            dialog.setTitle("Error Message");
            dialog.show();
        }
    
}
