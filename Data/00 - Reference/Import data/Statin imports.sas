data import_all;
 
*make sure variables to store file name are long enough;
length filename txt_file_name $256;
          format Entry_DOC_Line $21. ;
         format Arrival_Date $21. ;
         format Submission_Date $21. ;
         format Port_of_Entry_Distrct_Abrvtn $6. ;
         format Country_Of_Origin $4. ;
         format Product_Code $9. ;
         format Product_Code_Description $50. ;
         format Manufacturer_FEI_Number $25. ;
         format Manufacturer_Legal_Name $51. ;
         format Manufacturer_Line1_Address $150. ;
         format Manufacturer_Line2_Address $150. ;
         format Manufacturer_City_Name $50. ;
         format Manufacturer_ISO_Country_Code $4. ;
         format Filer_FEI_Number $25. ;
         format Filer_Legal_Name $50. ;
         format Filer_Line1_Address $150. ;
         format Filer_Line2_Address $150. ;
         format Filer_City_Name $50. ;
         format Filer_State_Code $4. ;
         format Filer_County_Code $5. ;
         format Filer_Zip_Code $12. ;
         format Filer_ISO_Country_Code $4. ;
         format Final_Disposition_Activity_Desc $19. ;
         format Final_Disposition_Activity_Date $21. ;
*keep file name from record to record;
retain txt_file_name;
 
*Use wildcard in input;
infile "C:\DB\FDA\Imports\*.csv" eov=eov filename=filename  DSD  truncover MISSOVER ;
 
*Input first record and hold line;
input@;
 
*Check if this is the first record or the first record in a new file;
*If it is, replace the filename with the new file name and move to next line;
if _n_ eq 1 or eov then do;
txt_file_name = scan(filename, -1, "\");
eov=0;delete;
end;
 
*Otherwise  go to the import step and read the files;

 
*Place input code here;
       input
                  Entry_DOC_Line $ 
          Arrival_Date $ 
          Submission_Date $ 
          Port_of_Entry_Distrct_Abrvtn $ 
          Country_Of_Origin $ 
          Product_Code $ 
          Product_Code_Description $ 
          Manufacturer_FEI_Number $ 
          Manufacturer_Legal_Name $ 
          Manufacturer_Line1_Address $ 
          Manufacturer_Line2_Address $ 
          Manufacturer_City_Name $ 
          Manufacturer_ISO_Country_Code $ 
          Filer_FEI_Number $ 
          Filer_Legal_Name $ 
          Filer_Line1_Address $ 
          Filer_Line2_Address $ 
          Filer_City_Name $ 
          Filer_State_Code $ 
          Filer_County_Code $ 
          Filer_Zip_Code $ 
          Filer_ISO_Country_Code $ 
          Final_Disposition_Activity_Desc $ 
          Final_Disposition_Activity_Date $ 
;
if prxmatch('/STATIN/i',upcase(Product_Code_Description))  then output;
run;


data statins;
set import_all;
if prxmatch('/ATORVASTATIN|CERIVASTATIN|FLUVASTATIN|LOVASTATIN|PITAVASTATIN|PITAVASTATIN|PITAVASTATIN|PRAVASTATIN|ROSUVASTATIN|ROSUVASTATIN|SIMVASTATIN/i',upcase(Product_Code_Description))  then output;
run;

proc sort data = statins out=x nodupkey;
by Product_Code_Description;
run;



data import_comm;
 
*make sure variables to store file name are long enough;
length filename txt_file_name $256;
          format Entry_DOC_Line $21. ;
         format Arrival_Date $21. ;
         format Submission_Date $21. ;
         format Port_of_Entry_Distrct_Abrvtn $6. ;
         format Country_Of_Origin $4. ;
         format Product_Code $9. ;
         format Product_Code_Description $50. ;
         format Manufacturer_FEI_Number $25. ;
         format Manufacturer_Legal_Name $51. ;
         format Manufacturer_Line1_Address $150. ;
         format Manufacturer_Line2_Address $150. ;
         format Manufacturer_City_Name $50. ;
         format Manufacturer_ISO_Country_Code $4. ;
         format Filer_FEI_Number $25. ;
         format Filer_Legal_Name $50. ;
         format Filer_Line1_Address $150. ;
         format Filer_Line2_Address $150. ;
         format Filer_City_Name $50. ;
         format Filer_State_Code $4. ;
         format Filer_County_Code $5. ;
         format Filer_Zip_Code $12. ;
         format Filer_ISO_Country_Code $4. ;
         format Final_Disposition_Activity_Desc $19. ;
         format Final_Disposition_Activity_Date $21. ;
*keep file name from record to record;
retain txt_file_name;
 
*Use wildcard in input;
infile "C:\DB\FDA\Imports\*.csv" eov=eov filename=filename  DSD  truncover MISSOVER ;
 
*Input first record and hold line;
input@;
 
*Check if this is the first record or the first record in a new file;
*If it is, replace the filename with the new file name and move to next line;
if _n_ eq 1 or eov then do;
txt_file_name = scan(filename, -1, "\");
eov=0;delete;
end;
 
*Otherwise  go to the import step and read the files;

 
       input
                  Entry_DOC_Line $ 
          Arrival_Date $ 
          Submission_Date $ 
          Port_of_Entry_Distrct_Abrvtn $ 
          Country_Of_Origin $ 
          Product_Code $ 
          Product_Code_Description $ 
          Manufacturer_FEI_Number $ 
          Manufacturer_Legal_Name $ 
          Manufacturer_Line1_Address $ 
          Manufacturer_Line2_Address $ 
          Manufacturer_City_Name $ 
          Manufacturer_ISO_Country_Code $ 
          Filer_FEI_Number $ 
          Filer_Legal_Name $ 
          Filer_Line1_Address $ 
          Filer_Line2_Address $ 
          Filer_City_Name $ 
          Filer_State_Code $ 
          Filer_County_Code $ 
          Filer_Zip_Code $ 
          Filer_ISO_Country_Code $ 
          Final_Disposition_Activity_Desc $ 
          Final_Disposition_Activity_Date $ 
;
if prxmatch('/LIPITOR|BAYCOL|LESCOL|MEVACOR|LIVALO|ZYPITAMAG|NIKITA|PRAVACHOL|CRESTOR|ZOCOR/i',upcase(Product_Code_Description))  then output;
run;

