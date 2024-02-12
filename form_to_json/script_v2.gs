// Edit URL. Note that this must be authenticated with the current user.
// format: https://docs.google.com/forms/d/{form-id}/edit
var URL = "https://docs.google.com/forms/d/1-HbK-Qnf6e44W0EyusihC5JucEBu7O1P1ToVDFMFiRs/edit";

function getFormMetadata(form){
  metadata = {
    "title": form.getTitle(),
    "id": form.getId(),
    "description": form.getDescription(),
    "publishedUrl": form.getPublishedUrl(),
    "editorEmails": form.getEditors().map(function(user) {return user.getEmail()}),
    "confirmationMessage": form.getConfirmationMessage(),
    "customClosedFormMessage": form.getCustomClosedFormMessage()
  }

  return metadata;

};

function convertToCamelCase(text){
  return text.toLowerCase().replace(/(\_\w)/g, function(m) {return m[1].toUpperCase();});
};

function main(){
  var form = FormApp.openByUrl(URL);
  var items = form.getItems();
  var count = items.length; // includes section

  var result = {
  "metadata": getFormMetadata(form),
  "count": count,
  "items": [],
  };

  // Iterate over all the items of the form
  items.forEach(function(value){
    var title = value.getTitle();
    var itemType = value.getType().toString();
    var index = value.getIndex();

    // initialize and empty dictionary to put all key value pairs of the item
    var data = {};

    // add key-value pairs to dictionary
    data["title"] = title;
    data["type"] = itemType;
    data["index"] = index;

    // need to cast item to a certain data type to perform type-specific function
    var itemTypeName = convertToCamelCase("AS_" + value.getType().toString() + "_ITEM")
    var typedItem = value[itemTypeName]();

    // get all keys of the item by filtering keys with "get", "is" and "has"
    var keys = Object.keys(typedItem).filter(function(s) {return s.indexOf("get") == 0});
    var boolKeys = Object.keys(typedItem).filter(function(s) {
    return (s.indexOf("is") == 0) || (s.indexOf("has") == 0) || (s.indexOf("includes") == 0);
  });

    // put all keys, boolKeys and their corresponding values to data dictionary
    keys.map(function(getKey){
      
      // convert all keys to function, skip if not a function
      try{
        var propName = getKey[3].toLowerCase() + getKey.substring(4);
        var propValue = typedItem[getKey]();
        data[propName] = propValue;
      } catch(e){
        // Logger.log(e);
      }
    });

    boolKeys.map(function(boolKey){
      // Logger.log(boolKey);
      var propName = boolKey;
      var propValue = typedItem[boolKey]();
      data[propName] = propValue;
    });

    // Get properties depending on type
    switch (value.getType()){
      // For dropdown and multiple choice, get the redirects
      case FormApp.ItemType.LIST:
      case FormApp.ItemType.MULTIPLE_CHOICE:
        data.choices = typedItem.getChoices().map(function(choice) {
          // Logger.log(Object.keys(choice))
          return {"choice":choice.getValue(),"navType":choice.getPageNavigationType(),"targetIndex":choice.getGotoPage().getIndex()};
        });
        break;
      case FormApp.ItemType.CHECKBOX:
        data.choices = typedItem.getChoices().map(function(choice) {
          return choice.getValue();
        });
        break;

      case FormApp.ItemType.IMAGE:
        data.alignment = typedItem.getAlignment().toString();
        
        if (item.getType() == FormApp.ItemType.VIDEO) {
          return;
        }
        
        var imageBlob = typedItem.getImage();
        
        data.imageBlob = {
          "dataAsString": imageBlob.getDataAsString(),
          "name": imageBlob.getName(),
          "isGoogleType": imageBlob.isGoogleType()
        };
        
        break;
      
    case FormApp.ItemType.PAGE_BREAK:
      data.pageNavigationType = typedItem.getPageNavigationType().toString();
      break;
      
    default:
      break;
    }

    result.items.push(data);
  });

  Logger.log(JSON.stringify(result));
}