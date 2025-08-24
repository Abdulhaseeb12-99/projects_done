package empmanager;

import javafx.scene.layout.GridPane;
import javafx.scene.layout.VBox;
import javafx.scene.layout.HBox;
import javafx.scene.control.TextField;
import javafx.scene.control.Label;
import javafx.scene.control.Button;

import javafx.scene.input.KeyCode;

import javafx.geometry.Insets;
import javafx.scene.Scene;
import javafx.stage.Stage;
import java.util.ArrayList;
import java.util.Arrays;
import javafx.geometry.Pos;
import javafx.stage.Modality;
import javafx.collections.ObservableList;




import java.sql.*;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.paint.Color;
import javafx.scene.text.Font;
import javafx.scene.text.Text;


/**
 *
 * @author ballaj
 */
public class AddEmployee {
    
    /**Create a private data to build the pane */
    private TextField tfFirstName=new TextField();
    private TextField tfMiddleName =new TextField();
    private TextField tfLastName=new TextField();
    private TextField tfPosition=new TextField();
    private TextField tfSalary=new TextField();
    
    private ArrayList<Label> labels= new ArrayList<Label>(Arrays.asList(
     new Label("First name: "), new Label("Middle name: "), new Label("Last name: "),
     new Label("Position: "), new Label("Salary: ")));
    
    private Button add=new Button("Add");
    private Button clear=new Button("Clear");
    private Button cancel =new Button ("Cancel");
    
    private GridPane gridPane =new GridPane();
    private VBox vbox =new VBox();
    private HBox hbox =new HBox();
    
    private Label title =new Label("Add Employee: ");
    
    /**Create a default constructor */
    public AddEmployee(Stage primaryStage, Statement stm, ObservableList<Employee> data, int [] startingPoint,ArrayList<Employee> employees )
    {
        
        title.setStyle("-fx-font-family:calibri;"+"-fx-font-size:17;");
        
        
        /**Set the text fields properties  */
        tfFirstName.setStyle("-fx-background:yellow;");
        tfFirstName.setPrefWidth(200);
        tfMiddleName.setStyle("-fx-background:yellow;");
        tfLastName.setStyle("-fx-background:yellow;");
        tfPosition.setStyle("-fx-background:yellow;");
        tfSalary.setStyle("-fx-background:yellow;");
        
        /**Set the label properties */
        for (int i=0; i<labels.size();i++)
        {
            labels.get(i).setStyle("-fx-text-fill:black;"+"-fx-font-family:calibri;"+"-fx-font-size:15;");
        }
        
        /**Set the button properties*/
        add.setPrefWidth(100);
        clear.setPrefWidth(100);
        cancel.setPrefWidth(100);
        
        /**Set the grid pane properties */
        gridPane.setHgap(10);
        gridPane.setVgap(10);
        gridPane.setAlignment(Pos.CENTER);
        
        gridPane.add(labels.get(0),0,0);
        gridPane.add(tfFirstName,1,0);
        gridPane.add(labels.get(1),0,1);
        gridPane.add(tfMiddleName,1,1);
        gridPane.add(labels.get(2),0,2);
        gridPane.add(tfLastName,1,2);
        gridPane.add(labels.get(3),0,3);
        gridPane.add(tfPosition,1,3);
        gridPane.add(labels.get(4), 0, 4);
        gridPane.add(tfSalary,1,4);
        gridPane.add(hbox,0,5);
        
        
        
        /**Set the HBox properties */
        hbox.setAlignment(Pos.CENTER);
        hbox.setSpacing(20);
        hbox.getChildren().addAll(add,clear,cancel);
        
        
        /**Set the VBox properties */
        vbox.setSpacing(5);
        vbox.setPadding(new Insets(10,15,0,0));
        vbox.getChildren().addAll(title,gridPane,hbox);
        
        /** Create a Scene */
        Scene scene=new Scene(vbox,400,300);
        final Stage mainStage=new Stage();
        mainStage.initModality(Modality.APPLICATION_MODAL);
        mainStage.initOwner(primaryStage);
        
        mainStage.setScene(scene);
        mainStage.setTitle("Add Employee");
        mainStage.show();
        
        
        
        /**Register and handle the event fired by the clear button */
        clear.setOnAction(e->{clearTextFields();});
        
        
        /**Register and handle the event fires by the exit button */
        cancel.setOnAction(e->{mainStage.close();});
        
        /**Register and handle */
        
            add.setOnAction(e->{
              try{
              addNewEmployee(primaryStage,stm,data, startingPoint,employees);
              }catch(Exception exception)
              {
                  popErrorMessage(mainStage,exception.getMessage());
              }
              
            });
            
       /**Enter Key */
       vbox.setOnKeyPressed(e->{
          if(e.getCode()==KeyCode.ENTER)
          {
              try{
                 addNewEmployee(primaryStage,stm,data, startingPoint,employees);
              }catch(Exception exception)
              {
                  popErrorMessage(mainStage,exception.getMessage());
              }
          }
       });
        
    }
    
    
    /**Clear text fields */
    public void clearTextFields ()
    {
        tfFirstName.clear();
        tfMiddleName.clear();
        tfLastName.clear();
        tfPosition.clear();
        tfSalary.clear();
    }
    
    
    /**Add new employee */
    public void addNewEmployee(Stage stage,Statement stm, ObservableList<Employee> data, int [] startingPoint, ArrayList<Employee> employees) throws Exception
    {
        if (tfFirstName.getText().equals(""))
        {
            tfFirstName.requestFocus();
        }
        else if (tfMiddleName.getText().equals(""))
        {
            tfMiddleName.requestFocus();
        }
        else if (tfLastName.getText().equals(""))
        {
            tfLastName.requestFocus();
        }
        else if(tfPosition.getText().equals(""))
        {
            tfPosition.requestFocus();
        }
        else if (tfSalary.getText().equals(""))
        {
            tfSalary.requestFocus();
        }
        else {           
             
        String query ="insert into employee (firstname, middlename, lastname, position, salary) values "
                + "('"+tfFirstName.getText()+"','"+tfMiddleName.getText()+"','"+tfLastName.getText()+"','"+tfPosition.getText()+"','"+tfSalary.getText()+"')";
        
        stm.executeUpdate(query);
        clearTextFields();
        
        infoDialog(stage,"Employee Was Added Successfully!!!");
        
         query ="select * from employee where id>'"+startingPoint[0]+"'";
               ResultSet resultSet=stm.executeQuery(query);
               
               while(resultSet.next())
               {
                   data.add(new Employee(
                    Integer.parseInt(resultSet.getString(1)),
                    resultSet.getString(2),
                    resultSet.getString(4),
                    resultSet.getString(3),
                    resultSet.getString(5),
                    resultSet.getDouble(6)
                   ));
                   
                    employees.add(new Employee(
                    Integer.parseInt(resultSet.getString(1)),
                    resultSet.getString(2),
                    resultSet.getString(4),
                    resultSet.getString(3),
                    resultSet.getString(5),
                    resultSet.getDouble(6)
                   ));
                   
                   startingPoint[0]=Integer.parseInt(resultSet.getString(1));
                   
               }
        }       
    }
    
    
   
    /**Create Information Dialog method */
       public void infoDialog (Stage primaryStage, String text1)
       {
           /**Create a Stage  */
            final Stage dialog=new Stage ();
            dialog.initModality(Modality.APPLICATION_MODAL);
            dialog.initOwner(primaryStage);
            
            /**Set text properties */
            Text localText=new Text();
            localText.setText(text1);
            localText.setFill(Color.BLUE);
            localText.setFont(Font.font("Times new roman",20));
            
            /**Set image properties */
            ImageView image =new ImageView(new Image("file:info.jpg"));
           
            /**Set dialog HBox properties */
            HBox dialogHbox=new HBox();
            dialogHbox.setSpacing(10);
            dialogHbox.getChildren().addAll(image,localText);
            HBox.setMargin(image,new Insets(0,0,0,10));
            /**Set VBox properties */
            VBox dialogVBox=new VBox();
            Button dialogOk=new Button("Ok");
            dialogVBox.setSpacing(10);
            dialogVBox.getChildren().addAll(dialogHbox,dialogOk);
            VBox.setMargin(dialogOk,new Insets(0,0,0,170));
            
            /**Set the button properties */
            dialogOk.setPrefWidth(100);
            
            /** exit the pop up window*/
            dialogOk.setOnAction(new EventHandler <ActionEvent>(){
            
               public void handle (ActionEvent event)
               {
                   dialog.close();
               }
                
            });
            
            /**Create a scene  */
            Scene scene=new Scene (dialogVBox,450,100);
            
            dialog.setScene(scene);
            dialog.setTitle("Info Message");
            dialog.show();
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
