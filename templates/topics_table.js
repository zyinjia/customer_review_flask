

function getTable()
{
  var data = [{ "subject":"ProgApps", "codeNo":"9594", "courseNo":"IT 312L", "instructor":"Maria Clemente Concepcion" },
              { "subject":"ITCR", "codeNo":"9615", "courseNo":"IT 014", "instructor":"Jonathan Ramirez" },
              { "subject":"ITP2", "codeNo":"9602", "courseNo":"IT 421", "instructor":"Jonathan Ramirez" }];
  var thisElement = document.getElementById('mytablecontainer');
      thisElement.innerHTML = "<table>";
      for (var x =0; x <len(data); x++){
          thisElement.innerHTML = thisElement.innerHTML + "<tr><td>"
                                  + data[x].subject +"</td> <td>"
                                  + data[x].codeNo +"</td></tr>";
      };
      thisElement.innerHTML = thisElement.innerHTML + "</table>";
};
