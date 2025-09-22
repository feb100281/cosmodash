// Здесь описываем функции и методы для ag grid


// Стороим конструкцию

var dagfuncs = (window.dashAgGridFunctions = window.dashAgGridFunctions || {});

dagfuncs.Intl = Intl;

dagfuncs.RussianDate = function (date, filna = "") {
    if (isNaN(date)) {
        return filna;        
    }
    return date;
};